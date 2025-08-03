package server

import (
	"context"
	"encoding/json"
	"fmt"
	"pi/util"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stellar/go/keypair"
)

type EnhancedWithdrawRequest struct {
	SeedPhrase        string                 `json:"seed_phrase"`
	LockedBalanceID   string                 `json:"locked_balance_id"`
	WithdrawalAddress string                 `json:"withdrawal_address"`
	Amount            string                 `json:"amount"`
	Strategy          map[string]interface{} `json:"strategy,omitempty"`
	Priority          string                 `json:"priority,omitempty"`
}

type TurboWithdrawRequest struct {
	EnhancedWithdrawRequest
	PreExecutionTime int    `json:"pre_execution_time_ms,omitempty"`
	BurstSize        int    `json:"burst_size,omitempty"`
	MaxRetries       int    `json:"max_retries,omitempty"`
	FeeStrategy      string `json:"fee_strategy,omitempty"`
}

type PerformanceResponse struct {
	Success bool                   `json:"success"`
	Message string                 `json:"message"`
	Stats   map[string]interface{} `json:"stats"`
}

func (s *Server) EnhancedWithdraw(ctx *gin.Context) {
	conn, err := upgrader.Upgrade(ctx.Writer, ctx.Request, nil)
	if err != nil {
		ctx.JSON(500, gin.H{"message": "Failed to upgrade to WebSocket"})
		return
	}
	defer func() {
		conn.Close()
	}()

	var req EnhancedWithdrawRequest
	_, message, err := conn.ReadMessage()
	if err != nil {
		s.sendEnhancedErrorResponse(conn, "Invalid request")
		return
	}

	err = json.Unmarshal(message, &req)
	if err != nil {
		s.sendEnhancedErrorResponse(conn, "Malformed JSON")
		return
	}

	kp, err := util.GetKeyFromSeed(req.SeedPhrase)
	if err != nil {
		s.sendEnhancedErrorResponse(conn, "invalid seed phrase")
		return
	}

	if req.Strategy != nil {
		if err := s.enhancedWallet.ConfigureStrategy(req.Strategy); err != nil {
			s.sendEnhancedErrorResponse(conn, fmt.Sprintf("Strategy configuration failed: %v", err))
			return
		}
	}

	s.executeEnhancedWithdraw(ctx, conn, kp, req)
}

func (s *Server) TurboWithdraw(ctx *gin.Context) {
	conn, err := upgrader.Upgrade(ctx.Writer, ctx.Request, nil)
	if err != nil {
		ctx.JSON(500, gin.H{"message": "Failed to upgrade to WebSocket"})
		return
	}
	defer func() {
		conn.Close()
	}()

	var req TurboWithdrawRequest
	_, message, err := conn.ReadMessage()
	if err != nil {
		s.sendEnhancedErrorResponse(conn, "Invalid request")
		return
	}

	err = json.Unmarshal(message, &req)
	if err != nil {
		s.sendEnhancedErrorResponse(conn, "Malformed JSON")
		return
	}

	kp, err := util.GetKeyFromSeed(req.SeedPhrase)
	if err != nil {
		s.sendEnhancedErrorResponse(conn, "invalid seed phrase")
		return
	}

	turboStrategy := map[string]interface{}{
		"aggressive_mode": true,
		"burst_size":      100,
		"max_retries":     1000,
		"fee_multiplier":  5.0,
	}

	if req.BurstSize > 0 {
		turboStrategy["burst_size"] = req.BurstSize
	}
	if req.MaxRetries > 0 {
		turboStrategy["max_retries"] = req.MaxRetries
	}

	if err := s.enhancedWallet.ConfigureStrategy(turboStrategy); err != nil {
		s.sendEnhancedErrorResponse(conn, fmt.Sprintf("Turbo strategy configuration failed: %v", err))
		return
	}

	s.sendEnhancedResponse(conn, WithdrawResponse{
		Action:  "turbo_mode_activated",
		Message: "Turbo mode activated with maximum performance settings",
		Success: true,
	})

	s.executeEnhancedWithdraw(ctx, conn, kp, req.EnhancedWithdrawRequest)
}

func (s *Server) executeEnhancedWithdraw(
	ctx context.Context,
	conn *websocket.Conn,
	kp *keypair.Full,
	req EnhancedWithdrawRequest,
) {
	s.withdrawAvailableBalance(conn, kp, req.WithdrawalAddress)

	balance, err := s.enhancedWallet.GetClaimableBalance(req.LockedBalanceID)
	if err != nil {
		s.sendEnhancedErrorResponse(conn, fmt.Sprintf("Failed to get balance: %v", err))
		return
	}

	var claimableAt time.Time
	var found bool
	
	for _, claimant := range balance.Claimants {
		if claimant.Destination == kp.Address() {
			claimableAt, found = util.ExtractClaimableTime(claimant.Predicate)
			if found {
				break
			}
		}
	}

	if !found {
		s.sendEnhancedErrorResponse(conn, "Unable to determine unlock time")
		return
	}

	s.sendEnhancedResponse(conn, WithdrawResponse{
		Action:  "scheduled",
		Message: fmt.Sprintf("Enhanced operations scheduled for execution at %v", claimableAt),
		Time:    claimableAt.Format(time.RFC3339),
		Success: true,
	})

	go s.sendCountdownUpdates(conn, claimableAt)

	err = s.enhancedWallet.ExecuteConcurrentClaimAndTransfer(
		ctx,
		kp,
		req.LockedBalanceID,
		req.WithdrawalAddress,
		req.Amount,
		claimableAt,
	)

	if err != nil {
		s.sendEnhancedResponse(conn, WithdrawResponse{
			Action:  "failed",
			Message: fmt.Sprintf("Enhanced execution failed: %v", err),
			Success: false,
		})
	} else {
		s.sendEnhancedResponse(conn, WithdrawResponse{
			Action:  "completed",
			Message: "Successfully executed concurrent claim and transfer with enhanced performance",
			Success: true,
		})
	}
}

func (s *Server) withdrawAvailableBalance(conn *websocket.Conn, kp *keypair.Full, withdrawalAddress string) {
	availableBalance, err := s.enhancedWallet.GetAvailableBalance(kp)
	if err != nil {
		s.sendEnhancedResponse(conn, WithdrawResponse{
			Action:  "available_balance_check_failed",
			Message: "Error checking available balance: " + err.Error(),
			Success: false,
		})
		return
	}

	balance, err := strconv.ParseFloat(availableBalance, 64)
	if err != nil || balance < 0.01 {
		s.sendEnhancedResponse(conn, WithdrawResponse{
			Action:  "no_available_balance",
			Message: fmt.Sprintf("No significant available balance to withdraw: %s PI", availableBalance),
			Success: true,
		})
		return
	}

	err = s.enhancedWallet.Transfer(kp, availableBalance, withdrawalAddress)
	if err != nil {
		s.sendEnhancedResponse(conn, WithdrawResponse{
			Action:  "available_balance_withdrawal_failed",
			Message: "Error withdrawing available balance: " + err.Error(),
			Success: false,
		})
	} else {
		s.sendEnhancedResponse(conn, WithdrawResponse{
			Action:  "available_balance_withdrawn",
			Message: fmt.Sprintf("Successfully withdrew available balance: %s PI", availableBalance),
			Amount:  balance,
			Success: true,
		})
	}
}

func (s *Server) sendCountdownUpdates(conn *websocket.Conn, targetTime time.Time) {
	ticker := time.NewTicker(time.Second * 10)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			remaining := time.Until(targetTime)
			if remaining <= 0 {
				return
			}

			s.sendEnhancedResponse(conn, WithdrawResponse{
				Action:  "countdown",
				Message: fmt.Sprintf("Time remaining: %v", remaining.Round(time.Second)),
				Success: true,
			})

			if remaining < time.Second*30 {
				return
			}
		}
	}
}

func (s *Server) GetPerformanceMetrics(ctx *gin.Context) {
	stats := s.enhancedWallet.GetPerformanceStats()
	
	ctx.JSON(200, PerformanceResponse{
		Success: true,
		Message: "Performance metrics retrieved successfully",
		Stats:   stats,
	})
}

func (s *Server) ConfigureStrategy(ctx *gin.Context) {
	var config map[string]interface{}
	
	if err := ctx.ShouldBindJSON(&config); err != nil {
		ctx.JSON(400, gin.H{
			"success": false,
			"message": fmt.Sprintf("Invalid configuration: %v", err),
		})
		return
	}

	if err := s.enhancedWallet.ConfigureStrategy(config); err != nil {
		ctx.JSON(400, gin.H{
			"success": false,
			"message": fmt.Sprintf("Configuration failed: %v", err),
		})
		return
	}

	ctx.JSON(200, gin.H{
		"success": true,
		"message": "Strategy configured successfully",
		"config": config,
	})
}

// Enhanced-specific response methods to avoid conflicts
func (s *Server) sendEnhancedErrorResponse(conn *websocket.Conn, message string) {
	s.sendEnhancedResponse(conn, WithdrawResponse{
		Action:  "error",
		Message: message,
		Success: false,
	})
}

func (s *Server) sendEnhancedResponse(conn *websocket.Conn, response WithdrawResponse) {
	writeMu.Lock()
	defer writeMu.Unlock()
	
	response.Time = time.Now().Format(time.RFC3339)
	conn.WriteJSON(response)
}