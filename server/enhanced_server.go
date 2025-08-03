package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"pi/util"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stellar/go/keypair"
)

// Enhanced withdraw request with additional options
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
	PreExecutionTime int    `json:"pre_execution_time_ms,omitempty"` // milliseconds before execution
	BurstSize        int    `json:"burst_size,omitempty"`
	MaxRetries       int    `json:"max_retries,omitempty"`
	FeeStrategy      string `json:"fee_strategy,omitempty"` // "aggressive", "ultra", "emergency"
}

type PerformanceResponse struct {
	Success bool                   `json:"success"`
	Message string                 `json:"message"`
	Stats   map[string]interface{} `json:"stats"`
}

// Enhanced withdraw handler with superior performance
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
		s.sendErrorResponse(conn, "Invalid request")
		return
	}

	err = json.Unmarshal(message, &req)
	if err != nil {
		s.sendErrorResponse(conn, "Malformed JSON")
		return
	}

	kp, err := util.GetKeyFromSeed(req.SeedPhrase)
	if err != nil {
		s.sendErrorResponse(conn, "invalid seed phrase")
		return
	}

	// Configure strategy if provided
	if req.Strategy != nil {
		if err := s.enhancedWallet.ConfigureStrategy(req.Strategy); err != nil {
			s.sendErrorResponse(conn, fmt.Sprintf("Strategy configuration failed: %v", err))
			return
		}
	}

	// Enhanced execution with concurrent claim and transfer
	s.executeEnhancedWithdraw(ctx, conn, kp, req)
}

// Turbo withdraw - maximum performance mode
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
		s.sendErrorResponse(conn, "Invalid request")
		return
	}

	err = json.Unmarshal(message, &req)
	if err != nil {
		s.sendErrorResponse(conn, "Malformed JSON")
		return
	}

	kp, err := util.GetKeyFromSeed(req.SeedPhrase)
	if err != nil {
		s.sendErrorResponse(conn, "invalid seed phrase")
		return
	}

	// Configure turbo strategy
	turboStrategy := map[string]interface{}{
		"aggressive_mode": true,
		"burst_size":      100,
		"max_retries":     1000,
		"fee_multiplier":  5.0,
	}

	// Override with user preferences
	if req.BurstSize > 0 {
		turboStrategy["burst_size"] = req.BurstSize
	}
	if req.MaxRetries > 0 {
		turboStrategy["max_retries"] = req.MaxRetries
	}

	if err := s.enhancedWallet.ConfigureStrategy(turboStrategy); err != nil {
		s.sendErrorResponse(conn, fmt.Sprintf("Turbo strategy configuration failed: %v", err))
		return
	}

	s.sendResponse(conn, WithdrawResponse{
		Action:  "turbo_mode_activated",
		Message: "Turbo mode activated with maximum performance settings",
		Success: true,
	})

	// Execute with turbo settings
	s.executeEnhancedWithdraw(ctx, conn, kp, req.EnhancedWithdrawRequest)
}

func (s *Server) executeEnhancedWithdraw(
	ctx context.Context,
	conn *websocket.Conn,
	kp *keypair.Full,
	req EnhancedWithdrawRequest,
) {
	// First, attempt to withdraw any available balance immediately
	s.withdrawAvailableBalance(conn, kp, req.WithdrawalAddress)

	// Get claimable balance details
	balance, err := s.enhancedWallet.GetClaimableBalance(req.LockedBalanceID)
	if err != nil {
		s.sendErrorResponse(conn, fmt.Sprintf("Failed to get balance: %v", err))
		return
	}

	// Find claimable time
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
		s.sendErrorResponse(conn, "Unable to determine unlock time")
		return
	}

	s.sendResponse(conn, WithdrawResponse{
		Action:  "scheduled",
		Message: fmt.Sprintf("Enhanced operations scheduled for execution at %v", claimableAt),
		Time:    claimableAt.Format(time.RFC3339),
		Success: true,
	})

	// Send countdown updates
	go s.sendCountdownUpdates(conn, claimableAt)

	// Execute concurrent operations with enhanced wallet
	err = s.enhancedWallet.ExecuteConcurrentClaimAndTransfer(
		ctx,
		kp,
		req.LockedBalanceID,
		req.WithdrawalAddress,
		req.Amount,
		claimableAt,
	)

	if err != nil {
		s.sendResponse(conn, WithdrawResponse{
			Action:  "failed",
			Message: fmt.Sprintf("Enhanced execution failed: %v", err),
			Success: false,
		})
	} else {
		s.sendResponse(conn, WithdrawResponse{
			Action:  "completed",
			Message: "Successfully executed concurrent claim and transfer with enhanced performance",
			Success: true,
		})
	}
}

func (s *Server) withdrawAvailableBalance(conn *websocket.Conn, kp *keypair.Full, withdrawalAddress string) {
	availableBalance, err := s.enhancedWallet.GetAvailableBalance(kp)
	if err != nil {
		s.sendResponse(conn, WithdrawResponse{
			Action:  "available_balance_check_failed",
			Message: "Error checking available balance: " + err.Error(),
			Success: false,
		})
		return
	}

	// Convert to float to check if there's a meaningful amount
	balance, err := strconv.ParseFloat(availableBalance, 64)
	if err != nil || balance < 0.01 { // Less than 0.01 PI
		s.sendResponse(conn, WithdrawResponse{
			Action:  "no_available_balance",
			Message: fmt.Sprintf("No significant available balance to withdraw: %s PI", availableBalance),
			Success: true,
		})
		return
	}

	err = s.enhancedWallet.Transfer(kp, availableBalance, withdrawalAddress)
	if err != nil {
		s.sendResponse(conn, WithdrawResponse{
			Action:  "available_balance_withdrawal_failed",
			Message: "Error withdrawing available balance: " + err.Error(),
			Success: false,
		})
	} else {
		s.sendResponse(conn, WithdrawResponse{
			Action:  "available_balance_withdrawn",
			Message: fmt.Sprintf("Successfully withdrew available balance: %s PI", availableBalance),
			Amount:  balance,
			Success: true,
		})
	}
}

func (s *Server) sendCountdownUpdates(conn *websocket.Conn, targetTime time.Time) {
	ticker := time.NewTicker(time.Second * 10) // Update every 10 seconds
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			remaining := time.Until(targetTime)
			if remaining <= 0 {
				return
			}

			s.sendResponse(conn, WithdrawResponse{
				Action:  "countdown",
				Message: fmt.Sprintf("Time remaining: %v", remaining.Round(time.Second)),
				Success: true,
			})

			// Stop sending updates when close to execution time
			if remaining < time.Second*30 {
				return
			}
		}
	}
}

// Performance metrics endpoint
func (s *Server) GetPerformanceMetrics(ctx *gin.Context) {
	stats := s.enhancedWallet.metrics.GetStats()
	
	ctx.JSON(200, PerformanceResponse{
		Success: true,
		Message: "Performance metrics retrieved successfully",
		Stats:   stats,
	})
}

// Strategy configuration endpoint
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

func (s *Server) sendErrorResponse(conn *websocket.Conn, message string) {
	s.sendResponse(conn, WithdrawResponse{
		Action:  "error",
		Message: message,
		Success: false,
	})
}

func (s *Server) sendResponse(conn *websocket.Conn, response WithdrawResponse) {
	writeMu.Lock()
	defer writeMu.Unlock()
	
	response.Time = time.Now().Format(time.RFC3339)
	conn.WriteJSON(response)
}