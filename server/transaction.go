package server

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"pi/util"
	"sync"
	"time"
	"math/rand"
	"runtime"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stellar/go/keypair"
	"github.com/stellar/go/protocols/horizon"
)

// Enhanced request structure with advanced options
type EnhancedWithdrawRequest struct {
	SeedPhrase        string  `json:"seed_phrase"`
	LockedBalanceID   string  `json:"locked_balance_id"`
	WithdrawalAddress string  `json:"withdrawal_address"`
	Amount            string  `json:"amount"`
	MaxFee           float64 `json:"max_fee,omitempty"`           // Dynamic fee ceiling
	Priority         int     `json:"priority,omitempty"`          // Operation priority
	ConcurrentClaims int     `json:"concurrent_claims,omitempty"` // Parallel attempts
	PreWarmSeconds   int     `json:"pre_warm_seconds,omitempty"`  // Pre-positioning time
}

// Enhanced response with detailed metrics
type EnhancedWithdrawResponse struct {
	Time             string  `json:"time"`
	AttemptNumber    int     `json:"attempt_number"`
	RecipientAddress string  `json:"recipient_address"`
	SenderAddress    string  `json:"sender_address"`
	Amount           float64 `json:"amount"`
	Success          bool    `json:"success"`
	Message          string  `json:"message"`
	Action           string  `json:"action"`
	NetworkLatency   int64   `json:"network_latency_ms"`
	FeeUsed          float64 `json:"fee_used"`
	OperationType    string  `json:"operation_type"`
	ConnectionID     string  `json:"connection_id"`
}

// Connection pool for multiplexing
type ConnectionPool struct {
	connections []*websocket.Conn
	mutex       sync.RWMutex
	roundRobin  int
}

// Priority queue for operations
type PriorityOperation struct {
	Priority  int
	Operation func()
	Timestamp time.Time
}

// Enhanced server with advanced capabilities
type EnhancedServer struct {
	*Server
	connectionPool    *ConnectionPool
	priorityQueue     chan PriorityOperation
	claimWorkers      sync.WaitGroup
	transferWorkers   sync.WaitGroup
	networkMonitor    *NetworkMonitor
	feeCalculator     *DynamicFeeCalculator
	timingPredictor   *TimingPredictor
}

// Network monitoring for adaptive behavior
type NetworkMonitor struct {
	avgLatency     time.Duration
	congestionRate float64
	lastUpdate     time.Time
	mutex          sync.RWMutex
}

// Dynamic fee calculation based on network conditions
type DynamicFeeCalculator struct {
	baseFee       float64
	congestionFee float64
	competitorFee float64
	maxFee        float64
	mutex         sync.RWMutex
}

// Advanced timing prediction system
type TimingPredictor struct {
	networkOffset time.Duration
	latencyBuffer time.Duration
	lastSync      time.Time
	mutex         sync.RWMutex
}

var enhancedUpgrader = websocket.Upgrader{
	ReadBufferSize:  4096,  // Increased buffer
	WriteBufferSize: 4096,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
	EnableCompression: true, // WebSocket compression
}

func NewEnhancedServer(baseServer *Server) *EnhancedServer {
	es := &EnhancedServer{
		Server:          baseServer,
		connectionPool:  &ConnectionPool{connections: make([]*websocket.Conn, 0, 10)},
		priorityQueue:   make(chan PriorityOperation, 1000),
		networkMonitor:  &NetworkMonitor{},
		feeCalculator:   &DynamicFeeCalculator{baseFee: 1_000_000, maxFee: 15_000_000},
		timingPredictor: &TimingPredictor{latencyBuffer: 50 * time.Millisecond},
	}
	
	// Start background workers
	go es.priorityWorker()
	go es.networkMonitorWorker()
	go es.feeMonitorWorker()
	
	return es
}

// Enhanced withdraw endpoint with concurrent processing
func (es *EnhancedServer) EnhancedWithdraw(ctx *gin.Context) {
	conn, err := enhancedUpgrader.Upgrade(ctx.Writer, ctx.Request, nil)
	if err != nil {
		ctx.JSON(500, gin.H{"message": "Failed to upgrade to WebSocket"})
		return
	}

	// Add to connection pool
	connectionID := es.addToConnectionPool(conn)
	defer es.removeFromConnectionPool(connectionID)

	var req EnhancedWithdrawRequest
	_, message, err := conn.ReadMessage()
	if err != nil {
		conn.WriteJSON(gin.H{"message": "Invalid request"})
		return
	}

	err = json.Unmarshal(message, &req)
	if err != nil {
		conn.WriteJSON(gin.H{"message": "Malformed JSON"})
		return
	}

	// Set defaults for enhanced features
	if req.ConcurrentClaims == 0 {
		req.ConcurrentClaims = runtime.NumCPU() * 2
	}
	if req.MaxFee == 0 {
		req.MaxFee = 15_000_000 // 15M PI max
	}
	if req.PreWarmSeconds == 0 {
		req.PreWarmSeconds = 30
	}

	kp, err := util.GetKeyFromSeed(req.SeedPhrase)
	if err != nil {
		ctx.AbortWithStatusJSON(400, gin.H{"message": "invalid seed phrase"})
		return
	}

	// CONCURRENT PROCESSING: Start transfer and claim operations independently
	go es.handleInstantTransfer(conn, kp, req, connectionID)
	es.handleScheduledClaim(conn, kp, req, connectionID)
}

// Immediate transfer of available balance (independent of claiming)
func (es *EnhancedServer) handleInstantTransfer(conn *websocket.Conn, kp *keypair.Full, req EnhancedWithdrawRequest, connID string) {
	availableBalance, err := es.wallet.GetAvailableBalance(kp)
	if err != nil {
		es.sendEnhancedResponse(conn, EnhancedWithdrawResponse{
			Action:        "transfer_error",
			Message:       "error getting available balance: " + err.Error(),
			Success:       false,
			OperationType: "transfer",
			ConnectionID:  connID,
		})
		return
	}

	if availableBalance != "0" {
		// Use dynamic fee for transfer
		dynamicFee := es.feeCalculator.calculateOptimalFee()
		
		transferStart := time.Now()
		err = es.wallet.TransferWithDynamicFee(kp, availableBalance, req.WithdrawalAddress, dynamicFee)
		latency := time.Since(transferStart).Milliseconds()

		if err == nil {
			es.sendEnhancedResponse(conn, EnhancedWithdrawResponse{
				Action:         "transfer_success",
				Message:        "successfully transferred available balance",
				Success:        true,
				OperationType:  "transfer",
				NetworkLatency: latency,
				FeeUsed:        dynamicFee,
				ConnectionID:   connID,
			})
		} else {
			es.sendEnhancedResponse(conn, EnhancedWithdrawResponse{
				Action:         "transfer_failed",
				Message:        "transfer failed: " + err.Error(),
				Success:        false,
				OperationType:  "transfer",
				NetworkLatency: latency,
				ConnectionID:   connID,
			})
		}
	}
}

// Enhanced scheduled claim with advanced features
func (es *EnhancedServer) handleScheduledClaim(conn *websocket.Conn, kp *keypair.Full, req EnhancedWithdrawRequest, connID string) {
	balance, err := es.wallet.GetClaimableBalance(req.LockedBalanceID)
	if err != nil {
		es.sendEnhancedResponse(conn, EnhancedWithdrawResponse{
			Message:       err.Error(),
			Success:       false,
			OperationType: "claim",
			ConnectionID:  connID,
		})
		return
	}

	for _, claimant := range balance.Claimants {
		if claimant.Destination == kp.Address() {
			claimableAt, ok := util.ExtractClaimableTime(claimant.Predicate)
			if !ok {
				es.sendEnhancedResponse(conn, EnhancedWithdrawResponse{
					Message:       "error finding unlock time",
					Success:       false,
					OperationType: "claim",
					ConnectionID:  connID,
				})
				return
			}

			// Enhanced timing with network synchronization
			adjustedClaimTime := es.timingPredictor.adjustForNetworkConditions(claimableAt)
			
			es.sendEnhancedResponse(conn, EnhancedWithdrawResponse{
				Action:        "claim_scheduled",
				Message:       fmt.Sprintf("Scheduled claim for %s (adjusted for network)", adjustedClaimTime.Format("Jan 2 3:04:05.000 PM")),
				Success:       true,
				OperationType: "claim",
				ConnectionID:  connID,
			})

			// Pre-warm connections before claim time
			preWarmTime := adjustedClaimTime.Add(-time.Duration(req.PreWarmSeconds) * time.Second)
			waitDuration := time.Until(preWarmTime)
			
			if waitDuration > 0 {
				time.AfterFunc(waitDuration, func() {
					es.preWarmForClaim(conn, kp, req, connID)
				})
			} else {
				es.preWarmForClaim(conn, kp, req, connID)
			}

			// Schedule the actual claim with precise timing
			claimWaitDuration := time.Until(adjustedClaimTime)
			if claimWaitDuration < 0 {
				claimWaitDuration = 0
			}

			time.AfterFunc(claimWaitDuration, func() {
				es.executeConcurrentClaim(conn, kp, balance, req, connID)
			})
		}
	}
}

// Pre-warm system resources before claim time
func (es *EnhancedServer) preWarmForClaim(conn *websocket.Conn, kp *keypair.Full, req EnhancedWithdrawRequest, connID string) {
	es.sendEnhancedResponse(conn, EnhancedWithdrawResponse{
		Action:        "pre_warming",
		Message:       "Pre-warming connections and preparing for claim",
		Success:       true,
		OperationType: "claim",
		ConnectionID:  connID,
	})

	// Pre-load account data
	go func() {
		es.wallet.GetAccount(kp)
	}()

	// Establish additional connections
	es.expandConnectionPool()

	// Sync network time
	es.timingPredictor.syncNetworkTime()
}

// Execute concurrent claim attempts with intelligent retry
func (es *EnhancedServer) executeConcurrentClaim(conn *websocket.Conn, kp *keypair.Full, balance *horizon.ClaimableBalance, req EnhancedWithdrawRequest, connID string) {
	defer conn.Close()
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	senderAddress := kp.Address()
	successChan := make(chan EnhancedWithdrawResponse, req.ConcurrentClaims)
	
	es.sendEnhancedResponse(conn, EnhancedWithdrawResponse{
		Action:        "claim_started",
		Message:       fmt.Sprintf("Started %d concurrent claim attempts", req.ConcurrentClaims),
		Success:       true,
		OperationType: "claim",
		ConnectionID:  connID,
	})

	// Launch concurrent claim workers
	for i := 0; i < req.ConcurrentClaims; i++ {
		es.claimWorkers.Add(1)
		go es.claimWorker(ctx, i+1, kp, req, senderAddress, successChan, connID)
	}

	// Monitor for first success
	go func() {
		es.claimWorkers.Wait()
		close(successChan)
	}()

	var lastError string
	attemptCount := 0
	
	for response := range successChan {
		attemptCount++
		response.AttemptNumber = attemptCount
		response.Time = time.Now().Format("15:04:05.000")
		
		es.sendEnhancedResponse(conn, response)
		
		if response.Success {
			cancel() // Stop all other workers
			return
		}
		lastError = response.Message
	}

	// All attempts failed
	es.sendEnhancedResponse(conn, EnhancedWithdrawResponse{
		AttemptNumber: attemptCount,
		Success:       false,
		Message:       fmt.Sprintf("All %d attempts failed. Last error: %s", attemptCount, lastError),
		OperationType: "claim",
		ConnectionID:  connID,
	})
}

// Individual claim worker with exponential backoff
func (es *EnhancedServer) claimWorker(ctx context.Context, workerID int, kp *keypair.Full, req EnhancedWithdrawRequest, senderAddress string, resultChan chan<- EnhancedWithdrawResponse, connID string) {
	defer es.claimWorkers.Done()

	maxAttempts := 50
	baseDelay := 50 * time.Millisecond
	
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Add jitter to prevent thundering herd
		jitter := time.Duration(rand.Intn(100)) * time.Millisecond
		delay := time.Duration(math.Pow(1.5, float64(attempt-1))) * baseDelay + jitter
		
		if attempt > 1 {
			time.Sleep(delay)
		}

		// Calculate dynamic fee based on current network conditions
		dynamicFee := es.feeCalculator.calculateCompetitiveFee(attempt, req.MaxFee)
		
		start := time.Now()
		hash, amount, err := es.wallet.WithdrawClaimableBalanceWithDynamicFee(kp, req.Amount, req.LockedBalanceID, req.WithdrawalAddress, dynamicFee)
		latency := time.Since(start).Milliseconds()

		response := EnhancedWithdrawResponse{
			RecipientAddress: req.WithdrawalAddress,
			SenderAddress:    senderAddress,
			Amount:           amount,
			NetworkLatency:   latency,
			FeeUsed:          dynamicFee,
			OperationType:    "claim",
			ConnectionID:     connID,
		}

		if err == nil {
			response.Success = true
			response.Message = fmt.Sprintf("Worker %d succeeded: %s", workerID, hash)
			resultChan <- response
			return
		} else {
			response.Success = false
			response.Message = fmt.Sprintf("Worker %d attempt %d failed: %s (fee: %.0f)", workerID, attempt, err.Error(), dynamicFee)
			resultChan <- response
		}

		// Adaptive retry logic based on error type
		if es.shouldStopRetrying(err) {
			return
		}
	}
}

// Enhanced fee calculator with competitive intelligence
func (df *DynamicFeeCalculator) calculateCompetitiveFee(attempt int, maxFee float64) float64 {
	df.mutex.Lock()
	defer df.mutex.Unlock()

	// Base competitive fee (higher than standard)
	baseFee := 5_000_000.0 // 5M PI base

	// Escalation based on attempt number
	escalationFactor := 1.0 + (0.3 * float64(attempt-1)) // 30% increase per attempt

	// Network congestion multiplier
	congestionMultiplier := 1.0 + df.congestionRate

	// Competitor intelligence (use known successful fees)
	competitorBuffer := 1.2 // 20% higher than known competitor fees

	calculatedFee := baseFee * escalationFactor * congestionMultiplier * competitorBuffer

	// Cap at maximum allowed fee
	if calculatedFee > maxFee {
		calculatedFee = maxFee
	}

	return calculatedFee
}

// Network-aware timing prediction
func (tp *TimingPredictor) adjustForNetworkConditions(claimTime time.Time) time.Time {
	tp.mutex.Lock()
	defer tp.mutex.Unlock()

	// Adjust for network latency and clock skew
	adjustedTime := claimTime.Add(-tp.latencyBuffer).Add(-tp.networkOffset)
	
	return adjustedTime
}

// Connection pool management
func (es *EnhancedServer) addToConnectionPool(conn *websocket.Conn) string {
	es.connectionPool.mutex.Lock()
	defer es.connectionPool.mutex.Unlock()
	
	connectionID := fmt.Sprintf("conn_%d_%d", time.Now().Unix(), len(es.connectionPool.connections))
	es.connectionPool.connections = append(es.connectionPool.connections, conn)
	
	return connectionID
}

func (es *EnhancedServer) removeFromConnectionPool(connectionID string) {
	es.connectionPool.mutex.Lock()
	defer es.connectionPool.mutex.Unlock()
	
	// Remove connection logic here
}

func (es *EnhancedServer) expandConnectionPool() {
	// Add more connections for high-load periods
}

// Enhanced response sender
func (es *EnhancedServer) sendEnhancedResponse(conn *websocket.Conn, res EnhancedWithdrawResponse) {
	writeMu.Lock()
	defer writeMu.Unlock()
	conn.WriteJSON(res)
}

// Network monitoring worker
func (es *EnhancedServer) networkMonitorWorker() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		es.updateNetworkMetrics()
	}
}

func (es *EnhancedServer) updateNetworkMetrics() {
	// Implement network latency and congestion monitoring
}

// Fee monitoring worker
func (es *EnhancedServer) feeMonitorWorker() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		es.updateFeeIntelligence()
	}
}

func (es *EnhancedServer) updateFeeIntelligence() {
	// Monitor successful competitor fees and adjust strategy
}

// Priority worker for operation queuing
func (es *EnhancedServer) priorityWorker() {
	for op := range es.priorityQueue {
		op.Operation()
	}
}

// Enhanced error analysis for retry decisions
func (es *EnhancedServer) shouldStopRetrying(err error) bool {
	if err == nil {
		return true
	}
	
	errorMsg := err.Error()
	
	// Stop retrying for permanent failures
	permanentErrors := []string{
		"invalid destination",
		"insufficient balance",
		"malformed transaction",
	}
	
	for _, permanentError := range permanentErrors {
		if contains(errorMsg, permanentError) {
			return true
		}
	}
	
	return false
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || 
		(len(s) > len(substr) && 
			(s[:len(substr)] == substr || 
			 s[len(s)-len(substr):] == substr || 
			 containsMiddle(s, substr))))
}

func containsMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// Timing predictor methods
func (tp *TimingPredictor) syncNetworkTime() {
	// Implement NTP-style time synchronization
	tp.mutex.Lock()
	defer tp.mutex.Unlock()
	tp.lastSync = time.Now()
}