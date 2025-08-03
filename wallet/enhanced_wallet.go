package wallet

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"pi/util"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/stellar/go/clients/horizonclient"
	hClient "github.com/stellar/go/clients/horizonclient"
	"github.com/stellar/go/keypair"
	"github.com/stellar/go/protocols/horizon"
	"github.com/stellar/go/txnbuild"
	"github.com/stellar/go/xdr"
)

// Enhanced Wallet with superior performance capabilities
type EnhancedWallet struct {
	*Wallet
	connectionPool    *ConnectionPool
	feeOracle        *DynamicFeeOracle
	timingSync       *NetworkTimingSync
	concurrencyMgr   *ConcurrencyManager
	floodProtection  *AntiFloodSystem
	operationCache   *OperationCache
	metrics          *PerformanceMetrics
	strategy         *ExecutionStrategy
}

// Connection pool for maximum throughput
type ConnectionPool struct {
	clients    []*hClient.Client
	current    int64
	maxClients int
	mu         sync.RWMutex
}

// Dynamic fee oracle for competitive bidding
type DynamicFeeOracle struct {
	baseFee          int64
	networkCongestion float64
	competitorFees   []int64
	mu               sync.RWMutex
	lastUpdate       time.Time
	feeHistory       []FeeEntry
	adaptiveMode     bool
}

type FeeEntry struct {
	Fee       int64
	Success   bool
	Timestamp time.Time
}

// Network timing synchronization for precise execution
type NetworkTimingSync struct {
	networkLatency   time.Duration
	clockSkew        time.Duration
	lastSync         time.Time
	syncInterval     time.Duration
	precisionBuffer  time.Duration
	ntpServers       []string
}

// Concurrency manager for parallel operations
type ConcurrencyManager struct {
	maxConcurrent int
	semaphore     chan struct{}
	activeOps     int64
	operationQueue chan *QueuedOperation
}

type QueuedOperation struct {
	ID       string
	Priority PriorityLevel
	Execute  func() error
	Result   chan error
}

// Anti-flood system to counter competitor flooding
type AntiFloodSystem struct {
	connectionBurst   int
	burstInterval     time.Duration
	backoffStrategy   BackoffStrategy
	floodDetection    *FloodDetector
	counterAttackMode bool
}

type BackoffStrategy int

const (
	ExponentialBackoff BackoffStrategy = iota
	LinearBackoff
	AdaptiveBackoff
	AggressiveBackoff
)

type FloodDetector struct {
	requestCounts    map[string]int
	timeWindows      map[string]time.Time
	threshold        int
	windowDuration   time.Duration
	mu               sync.RWMutex
	isUnderAttack    bool
}

// Operation cache for pre-computed transactions
type OperationCache struct {
	precomputedTxs map[string]*txnbuild.Transaction
	mu             sync.RWMutex
	ttl            time.Duration
	timestamps     map[string]time.Time
}

// Performance metrics and monitoring
type PerformanceMetrics struct {
	successRate      float64
	averageLatency   time.Duration
	operationCounts  map[string]int64
	competitorBeats  int64
	totalAttempts    int64
	mu               sync.RWMutex
	startTime        time.Time
}

// Execution strategy configuration
type ExecutionStrategy struct {
	AggressiveMode   bool
	BurstSize        int
	MaxRetries       int
	FeeMultiplier    float64
	PreExecutionTime time.Duration
	mu               sync.RWMutex
}

type PriorityLevel int

const (
	NormalPriority PriorityLevel = iota
	HighPriority
	UltraPriority
	EmergencyPriority
)

func NewEnhancedWallet() *EnhancedWallet {
	base := New()
	
	ew := &EnhancedWallet{
		Wallet:          base,
		connectionPool:  newConnectionPool(100), // Increased to 100 connections
		feeOracle:      newDynamicFeeOracle(),
		timingSync:     newNetworkTimingSync(),
		concurrencyMgr: newConcurrencyManager(200), // Increased concurrency
		floodProtection: newAntiFloodSystem(),
		operationCache: newOperationCache(),
		metrics:        newPerformanceMetrics(),
		strategy:       newExecutionStrategy(),
	}
	
	// Start background processes
	go ew.backgroundNetworkSync()
	go ew.backgroundFeeAnalysis()
	go ew.backgroundCacheCleanup()
	
	return ew
}

func newConnectionPool(maxClients int) *ConnectionPool {
	pool := &ConnectionPool{
		clients:    make([]*hClient.Client, maxClients),
		maxClients: maxClients,
	}
	
	// Initialize connection pool with staggered creation
	for i := 0; i < maxClients; i++ {
		client := &hClient.Client{
			HorizonURL: os.Getenv("NET_URL"),
		}
		// Configure for high performance
		client.HTTP.Timeout = time.Second * 30
		pool.clients[i] = client
	}
	
	return pool
}

func (cp *ConnectionPool) getClient() *hClient.Client {
	index := atomic.AddInt64(&cp.current, 1) % int64(cp.maxClients)
	return cp.clients[index]
}

func newDynamicFeeOracle() *DynamicFeeOracle {
	return &DynamicFeeOracle{
		baseFee:        100000, // 0.01 PI base
		competitorFees: make([]int64, 0, 1000),
		lastUpdate:     time.Now(),
		feeHistory:     make([]FeeEntry, 0, 10000),
		adaptiveMode:   true,
	}
}

func (dfo *DynamicFeeOracle) updateCompetitorFees(fees []int64) {
	dfo.mu.Lock()
	defer dfo.mu.Unlock()
	
	// Add to competitor fees
	dfo.competitorFees = append(dfo.competitorFees, fees...)
	
	// Keep only recent entries (last 1000)
	if len(dfo.competitorFees) > 1000 {
		dfo.competitorFees = dfo.competitorFees[len(dfo.competitorFees)-1000:]
	}
	
	dfo.lastUpdate = time.Now()
}

func (dfo *DynamicFeeOracle) calculateOptimalFee(priority PriorityLevel) int64 {
	dfo.mu.RLock()
	defer dfo.mu.RUnlock()
	
	// Base calculation
	fee := dfo.baseFee
	
	// Adaptive learning from history
	if dfo.adaptiveMode && len(dfo.feeHistory) > 10 {
		successfulFees := make([]int64, 0)
		for _, entry := range dfo.feeHistory {
			if entry.Success && time.Since(entry.Timestamp) < time.Hour {
				successfulFees = append(successfulFees, entry.Fee)
			}
		}
		
		if len(successfulFees) > 0 {
			sort.Slice(successfulFees, func(i, j int) bool {
				return successfulFees[i] < successfulFees[j]
			})
			// Use 75th percentile of successful fees
			percentile75 := successfulFees[int(float64(len(successfulFees))*0.75)]
			if percentile75 > fee {
				fee = percentile75
			}
		}
	}
	
	// Network congestion multiplier
	congestionMultiplier := 1.0 + dfo.networkCongestion
	
	// Competitor analysis
	if len(dfo.competitorFees) > 0 {
		sort.Slice(dfo.competitorFees, func(i, j int) bool {
			return dfo.competitorFees[i] > dfo.competitorFees[j]
		})
		
		// Beat top competitor fees with buffer
		if len(dfo.competitorFees) >= 3 {
			// Use 95th percentile of competitor fees
			percentile95 := dfo.competitorFees[int(float64(len(dfo.competitorFees))*0.05)]
			competitorBuffer := int64(float64(percentile95) * 1.5) // 50% buffer
			if competitorBuffer > fee {
				fee = competitorBuffer
			}
		}
	}
	
	// Priority-based multiplier
	switch priority {
	case HighPriority:
		fee = int64(float64(fee) * congestionMultiplier * 2.0)
	case UltraPriority:
		fee = int64(float64(fee) * congestionMultiplier * 5.0)
	case EmergencyPriority:
		fee = int64(float64(fee) * congestionMultiplier * 10.0)
	default:
		fee = int64(float64(fee) * congestionMultiplier)
	}
	
	// Ensure minimum competitive fee (based on your competitor data)
	minCompetitiveFee := int64(3200000) // 0.32 PI (your competitor's fee)
	if fee < minCompetitiveFee {
		fee = minCompetitiveFee
	}
	
	// For ultra-competitive scenarios, use emergency fee
	if priority >= UltraPriority {
		emergencyFee := int64(9400000) // 0.94 PI (highest competitor fee you mentioned)
		if fee < emergencyFee {
			fee = emergencyFee * 2 // Double the highest known competitor fee
		}
	}
	
	return fee
}

func (dfo *DynamicFeeOracle) recordFeeResult(fee int64, success bool) {
	dfo.mu.Lock()
	defer dfo.mu.Unlock()
	
	entry := FeeEntry{
		Fee:       fee,
		Success:   success,
		Timestamp: time.Now(),
	}
	
	dfo.feeHistory = append(dfo.feeHistory, entry)
	
	// Keep only recent entries
	if len(dfo.feeHistory) > 10000 {
		dfo.feeHistory = dfo.feeHistory[1000:]
	}
}

func newNetworkTimingSync() *NetworkTimingSync {
	return &NetworkTimingSync{
		syncInterval:    time.Second * 10, // More frequent sync
		precisionBuffer: time.Millisecond * 25, // Reduced buffer for more aggressive timing
		ntpServers: []string{
			"time.google.com",
			"time.cloudflare.com",
			"pool.ntp.org",
		},
	}
}

func (nts *NetworkTimingSync) syncWithNetwork(client *hClient.Client) error {
	var totalLatency time.Duration
	var samples int
	
	// Take multiple samples for better accuracy
	for i := 0; i < 5; i++ {
		start := time.Now()
		_, err := client.Ledgers(horizonclient.LedgerRequest{Limit: 1})
		if err != nil {
			continue
		}
		latency := time.Since(start)
		totalLatency += latency
		samples++
	}
	
	if samples > 0 {
		nts.networkLatency = totalLatency / time.Duration(samples) / 2
		nts.lastSync = time.Now()
	}
	
	return nil
}

func (nts *NetworkTimingSync) getOptimalExecutionTime(targetTime time.Time) time.Time {
	// Account for network latency and add precision buffer
	optimalTime := targetTime.Add(-nts.networkLatency).Add(-nts.precisionBuffer)
	
	// Ensure we don't execute too early
	now := time.Now()
	if optimalTime.Before(now) {
		optimalTime = now.Add(time.Millisecond * 10)
	}
	
	return optimalTime
}

func newConcurrencyManager(maxConcurrent int) *ConcurrencyManager {
	return &ConcurrencyManager{
		maxConcurrent:  maxConcurrent,
		semaphore:      make(chan struct{}, maxConcurrent),
		operationQueue: make(chan *QueuedOperation, maxConcurrent*2),
	}
}

func (cm *ConcurrencyManager) acquire() {
	cm.semaphore <- struct{}{}
	atomic.AddInt64(&cm.activeOps, 1)
}

func (cm *ConcurrencyManager) release() {
	<-cm.semaphore
	atomic.AddInt64(&cm.activeOps, -1)
}

func (cm *ConcurrencyManager) getActiveOperations() int64 {
	return atomic.LoadInt64(&cm.activeOps)
}

func newAntiFloodSystem() *AntiFloodSystem {
	return &AntiFloodSystem{
		connectionBurst:   50, // Increased burst size
		burstInterval:     time.Millisecond * 50, // Faster bursts
		backoffStrategy:   AggressiveBackoff,
		counterAttackMode: true,
		floodDetection: &FloodDetector{
			requestCounts:  make(map[string]int),
			timeWindows:    make(map[string]time.Time),
			threshold:      2000, // Higher threshold for detection
			windowDuration: time.Second * 5,
		},
	}
}

func newOperationCache() *OperationCache {
	return &OperationCache{
		precomputedTxs: make(map[string]*txnbuild.Transaction),
		ttl:            time.Minute * 5,
		timestamps:     make(map[string]time.Time),
	}
}

func newPerformanceMetrics() *PerformanceMetrics {
	return &PerformanceMetrics{
		operationCounts: make(map[string]int64),
		startTime:       time.Now(),
	}
}

func newExecutionStrategy() *ExecutionStrategy {
	return &ExecutionStrategy{
		AggressiveMode:   true,
		BurstSize:        100,
		MaxRetries:       500, // Increased retries
		FeeMultiplier:    2.0,
		PreExecutionTime: time.Millisecond * 200,
	}
}

// Superior concurrent claim and transfer execution
func (ew *EnhancedWallet) ExecuteConcurrentClaimAndTransfer(
	ctx context.Context,
	kp *keypair.Full,
	balanceID string,
	transferAddress string,
	amount string,
	executionTime time.Time,
) error {
	
	// Pre-synchronize network timing
	if err := ew.timingSync.syncWithNetwork(ew.connectionPool.getClient()); err != nil {
		return fmt.Errorf("timing sync failed: %w", err)
	}
	
	// Calculate optimal execution time
	optimalTime := ew.timingSync.getOptimalExecutionTime(executionTime)
	
	// Pre-compute transactions for maximum speed
	claimTx, transferTx, err := ew.precomputeTransactions(kp, balanceID, transferAddress, amount)
	if err != nil {
		return fmt.Errorf("precomputation failed: %w", err)
	}
	
	// Launch counter-flood attack if enabled
	if ew.floodProtection.counterAttackMode {
		go ew.launchCounterFloodAttack(ctx, executionTime)
	}
	
	// Wait until optimal execution time
	waitDuration := time.Until(optimalTime)
	if waitDuration > 0 {
		timer := time.NewTimer(waitDuration)
		defer timer.Stop()
		
		select {
		case <-timer.C:
			// Execute both operations concurrently
			return ew.executeConcurrentOperations(ctx, claimTx, transferTx, kp)
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	
	// Execute immediately if time has passed
	return ew.executeConcurrentOperations(ctx, claimTx, transferTx, kp)
}

func (ew *EnhancedWallet) precomputeTransactions(
	kp *keypair.Full,
	balanceID string,
	transferAddress string,
	amount string,
) (*txnbuild.Transaction, *txnbuild.Transaction, error) {
	
	// Get account for sequence number
	account, err := ew.GetAccount(kp)
	if err != nil {
		return nil, nil, err
	}
	
	sourceAccount := &txnbuild.SimpleAccount{
		AccountID: kp.Address(),
		Sequence:  account.SequenceNumber,
	}
	
	// Pre-compute claim transaction
	claimOp := &txnbuild.ClaimClaimableBalance{
		BalanceID: balanceID,
	}
	
	claimFee := ew.feeOracle.calculateOptimalFee(EmergencyPriority)
	claimTx, err := txnbuild.NewTransaction(
		txnbuild.TransactionParams{
			SourceAccount:        sourceAccount,
			IncrementSequenceNum: true,
			Operations:           []txnbuild.Operation{claimOp},
			BaseFee:              claimFee,
			Preconditions:        txnbuild.Preconditions{TimeBounds: txnbuild.NewInfiniteTimeout()},
		},
	)
	if err != nil {
		return nil, nil, err
	}
	
	// Pre-compute transfer transaction with incremented sequence
	transferSourceAccount := &txnbuild.SimpleAccount{
		AccountID: kp.Address(),
		Sequence:  account.SequenceNumber + 1,
	}
	
	transferOp := &txnbuild.Payment{
		Destination: transferAddress,
		Amount:      amount,
		Asset:       txnbuild.NativeAsset{},
	}
	
	transferFee := ew.feeOracle.calculateOptimalFee(EmergencyPriority)
	transferTx, err := txnbuild.NewTransaction(
		txnbuild.TransactionParams{
			SourceAccount:        transferSourceAccount,
			IncrementSequenceNum: true,
			Operations:           []txnbuild.Operation{transferOp},
			BaseFee:              transferFee,
			Preconditions:        txnbuild.Preconditions{TimeBounds: txnbuild.NewInfiniteTimeout()},
		},
	)
	if err != nil {
		return nil, nil, err
	}
	
	return claimTx, transferTx, nil
}

func (ew *EnhancedWallet) executeConcurrentOperations(
	ctx context.Context,
	claimTx *txnbuild.Transaction,
	transferTx *txnbuild.Transaction,
	kp *keypair.Full,
) error {
	
	var wg sync.WaitGroup
	var claimErr, transferErr error
	
	// Execute claim operation with burst strategy
	wg.Add(1)
	go func() {
		defer wg.Done()
		claimErr = ew.executeWithBurstStrategy(ctx, claimTx, kp, "claim")
	}()
	
	// Execute transfer operation (independent of claim result) with burst strategy
	wg.Add(1)
	go func() {
		defer wg.Done()
		transferErr = ew.executeWithBurstStrategy(ctx, transferTx, kp, "transfer")
	}()
	
	wg.Wait()
	
	// Record results
	ew.feeOracle.recordFeeResult(claimTx.BaseFee(), claimErr == nil)
	ew.feeOracle.recordFeeResult(transferTx.BaseFee(), transferErr == nil)
	
	// Return combined results
	if claimErr != nil && transferErr != nil {
		ew.metrics.recordFailure("both")
		return fmt.Errorf("both operations failed - claim: %v, transfer: %v", claimErr, transferErr)
	} else if claimErr != nil {
		ew.metrics.recordPartialSuccess("transfer_only")
		return fmt.Errorf("claim failed but transfer succeeded: %v", claimErr)
	} else if transferErr != nil {
		ew.metrics.recordPartialSuccess("claim_only")
		return fmt.Errorf("transfer failed but claim succeeded: %v", transferErr)
	}
	
	ew.metrics.recordSuccess("both")
	return nil
}

func (ew *EnhancedWallet) executeWithBurstStrategy(
	ctx context.Context,
	tx *txnbuild.Transaction,
	kp *keypair.Full,
	operation string,
) error {
	
	maxAttempts := ew.strategy.MaxRetries
	burstSize := ew.strategy.BurstSize
	
	var lastErr error
	
	// Execute in bursts for maximum probability of success
	for burst := 0; burst < maxAttempts/burstSize; burst++ {
		var burstWg sync.WaitGroup
		var successChan = make(chan struct{}, 1)
		var burstErrors = make([]error, burstSize)
		
		// Launch burst of concurrent attempts
		for i := 0; i < burstSize; i++ {
			burstWg.Add(1)
			go func(attemptIndex int) {
				defer burstWg.Done()
				
				select {
				case <-successChan:
					// Another goroutine succeeded, exit early
					return
				default:
				}
				
				ew.concurrencyMgr.acquire()
				defer ew.concurrencyMgr.release()
				
				// Use different client for each attempt
				client := ew.connectionPool.getClient()
				
				// Sign transaction fresh each time
				signedTx, err := tx.Sign(ew.networkPassphrase, kp)
				if err != nil {
					burstErrors[attemptIndex] = err
					return
				}
				
				// Submit transaction
				_, err = client.SubmitTransaction(signedTx)
				if err != nil {
					burstErrors[attemptIndex] = err
					return
				}
				
				// Signal success
				select {
				case successChan <- struct{}{}:
					// Successfully signaled
				default:
					// Channel already has success signal
				}
			}(i)
		}
		
		// Wait for burst completion or success
		done := make(chan struct{})
		go func() {
			burstWg.Wait()
			close(done)
		}()
		
		select {
		case <-successChan:
			// Success! Record and return
			ew.metrics.recordSuccess(operation)
			return nil
		case <-done:
			// All attempts in burst failed
			for _, err := range burstErrors {
				if err != nil {
					lastErr = err
					break
				}
			}
		case <-ctx.Done():
			return ctx.Err()
		}
		
		// Brief pause between bursts (adaptive backoff)
		if burst < maxAttempts/burstSize-1 {
			backoffDuration := ew.calculateAggressiveBackoff(burst, lastErr)
			select {
			case <-time.After(backoffDuration):
				continue
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	
	ew.metrics.recordFailure(operation)
	return fmt.Errorf("operation %s failed after %d burst attempts: %v", operation, maxAttempts, lastErr)
}

func (ew *EnhancedWallet) calculateAggressiveBackoff(attempt int, err error) time.Duration {
	if err == nil {
		return time.Millisecond * 10
	}
	
	base := time.Millisecond * 5 // Very aggressive base
	
	// Error-specific backoff
	errorString := strings.ToLower(err.Error())
	switch {
	case strings.Contains(errorString, "rate") || strings.Contains(errorString, "limit"):
		// Rate limiting - very short backoff
		return base
	case strings.Contains(errorString, "sequence"):
		// Sequence issues - immediate retry
		return time.Millisecond
	case strings.Contains(errorString, "insufficient"):
		// Balance issues - short backoff
		return base * 2
	case strings.Contains(errorString, "timeout"):
		// Network issues - medium backoff
		return base * 3
	default:
		// Unknown error - short exponential backoff
		backoff := base * time.Duration(math.Pow(1.2, float64(attempt)))
		if backoff > time.Second {
			backoff = time.Second
		}
		return backoff
	}
}

func (ew *EnhancedWallet) launchCounterFloodAttack(ctx context.Context, targetTime time.Time) {
	// Launch coordinated burst attack 500ms before target time
	burstTime := targetTime.Add(-time.Millisecond * 500)
	
	waitDuration := time.Until(burstTime)
	if waitDuration <= 0 {
		return // Too late
	}
	
	timer := time.NewTimer(waitDuration)
	defer timer.Stop()
	
	select {
	case <-timer.C:
		// Launch sustained connection bursts to reserve network capacity
		var wg sync.WaitGroup
		
		for wave := 0; wave < 5; wave++ {
			for i := 0; i < ew.floodProtection.connectionBurst; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					client := ew.connectionPool.getClient()
					// Keep connection warm with lightweight request
					client.Ledgers(horizonclient.LedgerRequest{Limit: 1})
				}()
			}
			
			// Small delay between waves
			time.Sleep(ew.floodProtection.burstInterval)
		}
		
		wg.Wait()
	case <-ctx.Done():
		return
	}
}

// Background processes for continuous optimization
func (ew *EnhancedWallet) backgroundNetworkSync() {
	ticker := time.NewTicker(ew.timingSync.syncInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			ew.timingSync.syncWithNetwork(ew.connectionPool.getClient())
		}
	}
}

func (ew *EnhancedWallet) backgroundFeeAnalysis() {
	ticker := time.NewTicker(time.Minute * 5)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			ew.analyzePastTransactionsForFeeOptimization()
		}
	}
}

func (ew *EnhancedWallet) backgroundCacheCleanup() {
	ticker := time.NewTicker(time.Minute * 10)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			ew.operationCache.cleanup()
		}
	}
}

func (ew *EnhancedWallet) analyzePastTransactionsForFeeOptimization() {
	// Implementation for analyzing network transactions to optimize fee strategy
	// This would analyze recent transactions to understand fee patterns
}

func (oc *OperationCache) cleanup() {
	oc.mu.Lock()
	defer oc.mu.Unlock()
	
	now := time.Now()
	for key, timestamp := range oc.timestamps {
		if now.Sub(timestamp) > oc.ttl {
			delete(oc.precomputedTxs, key)
			delete(oc.timestamps, key)
		}
	}
}

func (pm *PerformanceMetrics) recordSuccess(operation string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.operationCounts[operation+"_success"]++
	pm.totalAttempts++
}

func (pm *PerformanceMetrics) recordFailure(operation string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.operationCounts[operation+"_failure"]++
	pm.totalAttempts++
}

func (pm *PerformanceMetrics) recordPartialSuccess(operation string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.operationCounts[operation+"_partial"]++
	pm.totalAttempts++
}

func (pm *PerformanceMetrics) GetStats() map[string]interface{} {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	
	totalSuccess := pm.operationCounts["claim_success"] + pm.operationCounts["transfer_success"] + pm.operationCounts["both_success"]
	successRate := 0.0
	if pm.totalAttempts > 0 {
		successRate = float64(totalSuccess) / float64(pm.totalAttempts) * 100
	}
	
	return map[string]interface{}{
		"success_rate":      successRate,
		"total_attempts":    pm.totalAttempts,
		"operation_counts":  pm.operationCounts,
		"competitor_beats":  pm.competitorBeats,
		"uptime":           time.Since(pm.startTime).String(),
	}
}

// Configuration methods
func (ew *EnhancedWallet) ConfigureStrategy(config map[string]interface{}) error {
	ew.strategy.mu.Lock()
	defer ew.strategy.mu.Unlock()
	
	if val, ok := config["aggressive_mode"].(bool); ok {
		ew.strategy.AggressiveMode = val
	}
	if val, ok := config["burst_size"].(int); ok {
		ew.strategy.BurstSize = val
	}
	if val, ok := config["max_retries"].(int); ok {
		ew.strategy.MaxRetries = val
	}
	if val, ok := config["fee_multiplier"].(float64); ok {
		ew.strategy.FeeMultiplier = val
	}
	
	return nil
}