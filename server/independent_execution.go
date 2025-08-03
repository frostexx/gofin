// CRITICAL: Claim and Transfer work INDEPENDENTLY and CONCURRENTLY

func (s *Server) ExecuteIndependentClaimAndTransfer(kp *keypair.Full, req WithdrawRequest) {
	var wg sync.WaitGroup
	
	// Track 1: INDEPENDENT CLAIMING (Does NOT affect transfer)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for attempt := 1; attempt <= 100; attempt++ {
			// Claim attempts continue regardless of transfer status
			s.attemptClaim(kp, req.LockedBalanceID)
			time.Sleep(100 * time.Millisecond)
		}
	}()
	
	// Track 2: INDEPENDENT TRANSFER (Does NOT wait for claim)
	wg.Add(1) 
	go func() {
		defer wg.Done()
		for attempt := 1; attempt <= 100; attempt++ {
			// Transfer attempts continue regardless of claim status
			s.attemptTransfer(kp, req.WithdrawalAddress)
			time.Sleep(100 * time.Millisecond)
		}
	}()
	
	wg.Wait() // Both tracks run until success or max attempts
}