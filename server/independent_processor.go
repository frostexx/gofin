package server

import (
	"context"
	"sync"
	"time"
)

type IndependentProcessor struct {
	claimEngine    *ClaimEngine
	transferEngine *TransferEngine
	monitoring     *ConcurrentMonitoring
}

type ClaimEngine struct {
	workers    []*ClaimWorker
	strategies []ClaimStrategy
	mutex      sync.RWMutex
}

type TransferEngine struct {
	workers    []*TransferWorker  
	strategies []TransferStrategy
	mutex      sync.RWMutex
}

func (s *Server) ExecuteIndependentOperations(kp *keypair.Full, req WithdrawRequest, unlockTime time.Time) {
	processor := &IndependentProcessor{
		claimEngine:    NewClaimEngine(25),    // 25 claim workers
		transferEngine: NewTransferEngine(25), // 25 transfer workers
		monitoring:     NewConcurrentMonitoring(),
	}
	
	// Calculate precise execution time (500ms early for network latency)
	executeTime := unlockTime.Add(-500 * time.Millisecond)
	
	// Wait until execution time
	time.Sleep(time.Until(executeTime))
	
	// Execute BOTH operations SIMULTANEOUSLY and INDEPENDENTLY
	var wg sync.WaitGroup
	
	// INDEPENDENT CLAIM TRACK - Runs regardless of transfer status
	wg.Add(1)
	go func() {
		defer wg.Done()
		processor.ExecuteIndependentClaiming(kp, req)
	}()
	
	// INDEPENDENT TRANSFER TRACK - Runs regardless of claim status  
	wg.Add(1)
	go func() {
		defer wg.Done()
		processor.ExecuteIndependentTransfer(kp, req)
	}()
	
	// MONITORING TRACK - Reports on both independently
	wg.Add(1)
	go func() {
		defer wg.Done()
		processor.MonitorBothTracks(req.LockedBalanceID)
	}()
	
	wg.Wait()
}

func (ip *IndependentProcessor) ExecuteIndependentClaiming(kp *keypair.Full, req WithdrawRequest) {
	// Claim operations - completely independent
	for attempt := 1; attempt <= 100; attempt++ {
		// Multiple concurrent claim strategies
		var claimWG sync.WaitGroup
		
		// Strategy 1: Low fee rapid fire
		claimWG.Add(1)
		go func() {
			defer claimWG.Done()
			ip.claimEngine.AttemptLowFeeClaim(kp, req.LockedBalanceID)
		}()
		
		// Strategy 2: High fee priority  
		claimWG.Add(1)
		go func() {
			defer claimWG.Done()
			ip.claimEngine.AttemptHighFeeClaim(kp, req.LockedBalanceID)
		}()
		
		// Strategy 3: Maximum fee aggressive
		claimWG.Add(1)
		go func() {
			defer claimWG.Done()
			ip.claimEngine.AttemptMaxFeeClaim(kp, req.LockedBalanceID)
		}()
		
		claimWG.Wait()
		
		// Check if successful, continue if not
		time.Sleep(50 * time.Millisecond) // Brief pause between attempts
	}
}

func (ip *IndependentProcessor) ExecuteIndependentTransfer(kp *keypair.Full, req WithdrawRequest) {
	// Transfer operations - completely independent
	for attempt := 1; attempt <= 100; attempt++ {
		// Multiple concurrent transfer strategies
		var transferWG sync.WaitGroup
		
		// Strategy 1: Available balance transfer
		transferWG.Add(1)
		go func() {
			defer transferWG.Done()
			ip.transferEngine.AttemptAvailableTransfer(kp, req.WithdrawalAddress)
		}()
		
		// Strategy 2: Future balance prediction transfer
		transferWG.Add(1)
		go func() {
			defer transferWG.Done()
			ip.transferEngine.AttemptPredictiveTransfer(kp, req.WithdrawalAddress, req.Amount)
		}()
		
		transferWG.Wait()
		
		// Continue regardless of claim status
		time.Sleep(50 * time.Millisecond)
	}
}