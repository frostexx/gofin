package wallet

import (
	"fmt"
	"math/rand"
	"os"
	"pi/util"
	"strconv"
	"sync"
	"time"

	"github.com/stellar/go/clients/horizonclient"
	hClient "github.com/stellar/go/clients/horizonclient"
	"github.com/stellar/go/keypair"
	"github.com/stellar/go/protocols/horizon"
	"github.com/stellar/go/protocols/horizon/operations"
)

type Wallet struct {
	networkPassphrase string
	serverURL         string
	client            *hClient.Client
	baseReserve       float64
	baseReserveMutex  sync.RWMutex
	baseReserveTime   time.Time
	connectionPool    *ConnectionPool
	retryConfig       *RetryConfig
	feeStrategy       *DynamicFeeStrategy
}

type ConnectionPool struct {
	clients []*hClient.Client
	current int
	mutex   sync.Mutex
	size    int
}

type RetryConfig struct {
	MaxAttempts     int
	BaseDelay       time.Duration
	MaxDelay        time.Duration
	BackoffMultiple float64
	Jitter          bool
}

type DynamicFeeStrategy struct {
	baseFee     int64
	priorityFee int64
	maxFee      int64
	mutex       sync.RWMutex
}

func NewConnectionPool(size int, serverURL string) *ConnectionPool {
	pool := &ConnectionPool{
		clients: make([]*hClient.Client, size),
		size:    size,
	}
	
	for i := 0; i < size; i++ {
		client := hClient.DefaultPublicNetClient
		client.HorizonURL = serverURL
		pool.clients[i] = client
	}
	
	return pool
}

func (cp *ConnectionPool) GetClient() *hClient.Client {
	cp.mutex.Lock()
	defer cp.mutex.Unlock()
	
	client := cp.clients[cp.current]
	cp.current = (cp.current + 1) % cp.size
	return client
}

func NewDynamicFeeStrategy() *DynamicFeeStrategy {
	return &DynamicFeeStrategy{
		baseFee:     100000,    // 0.01 PI base
		priorityFee: 3200000,   // Competitor's successful fee
		maxFee:      9400000,   // Maximum observed competitive fee
	}
}

func (dfs *DynamicFeeStrategy) GetOptimalFee(networkCongestion float64, priority bool) int64 {
	dfs.mutex.RLock()
	defer dfs.mutex.RUnlock()
	
	if priority {
		// For high-priority claims/withdrawals, use escalating fees
		if networkCongestion > 0.8 {
			return dfs.maxFee
		} else if networkCongestion > 0.5 {
			return dfs.priorityFee
		}
	}
	
	// Calculate dynamic fee based on network conditions
	dynamicFee := int64(float64(dfs.baseFee) * (1 + networkCongestion*10))
	if dynamicFee > dfs.maxFee {
		return dfs.maxFee
	}
	
	return dynamicFee
}

func New() *Wallet {
	serverURL := os.Getenv("NET_URL")
	
	w := &Wallet{
		networkPassphrase: os.Getenv("NET_PASSPHRASE"),
		serverURL:         serverURL,
		connectionPool:    NewConnectionPool(10, serverURL), // Pool of 10 connections
		baseReserve:       0.49,
		baseReserveTime:   time.Time{},
		retryConfig: &RetryConfig{
			MaxAttempts:     50,
			BaseDelay:       100 * time.Millisecond,
			MaxDelay:        30 * time.Second,
			BackoffMultiple: 1.5,
			Jitter:          true,
		},
		feeStrategy: NewDynamicFeeStrategy(),
	}
	
	w.client = w.connectionPool.GetClient()
	go w.periodicBaseReserveUpdate()
	
	return w
}

func (w *Wallet) periodicBaseReserveUpdate() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		w.GetBaseReserve()
	}
}

func (w *Wallet) GetBaseReserve() {
	w.baseReserveMutex.Lock()
	defer w.baseReserveMutex.Unlock()
	
	// Cache for 5 minutes
	if time.Since(w.baseReserveTime) < 5*time.Minute && w.baseReserve > 0 {
		return
	}
	
	client := w.connectionPool.GetClient()
	ledger, err := client.Ledgers(horizonclient.LedgerRequest{
		Order: horizonclient.OrderDesc, 
		Limit: 1,
	})
	if err != nil {
		fmt.Printf("Error fetching base reserve: %v\n", err)
		return
	}

	if len(ledger.Embedded.Records) == 0 {
		return
	}

	baseReserveStr := ledger.Embedded.Records[0].BaseReserve
	w.baseReserve = float64(baseReserveStr) / 1e7
	w.baseReserveTime = time.Now()
}

func (w *Wallet) GetCachedBaseReserve() float64 {
	w.baseReserveMutex.RLock()
	defer w.baseReserveMutex.RUnlock()
	return w.baseReserve
}

func (w *Wallet) GetAddress(kp *keypair.Full) string {
	return kp.Address()
}

func (w *Wallet) Login(seedPhrase string) (*keypair.Full, error) {
	kp, err := util.GetKeyFromSeed(seedPhrase)
	if err != nil {
		return nil, err
	}

	return kp, nil
}

func (w *Wallet) GetAccountWithRetry(kp *keypair.Full) (horizon.Account, error) {
	var account horizon.Account
	var lastErr error
	
	for attempt := 0; attempt < w.retryConfig.MaxAttempts; attempt++ {
		client := w.connectionPool.GetClient()
		accReq := hClient.AccountRequest{AccountID: kp.Address()}
		
		account, lastErr = client.AccountDetail(accReq)
		if lastErr == nil {
			return account, nil
		}
		
		if attempt < w.retryConfig.MaxAttempts-1 {
			delay := w.calculateBackoffDelay(attempt)
			time.Sleep(delay)
		}
	}
	
	return horizon.Account{}, fmt.Errorf("failed after %d attempts: %v", w.retryConfig.MaxAttempts, lastErr)
}

func (w *Wallet) GetAccount(kp *keypair.Full) (horizon.Account, error) {
	return w.GetAccountWithRetry(kp)
}

func (w *Wallet) calculateBackoffDelay(attempt int) time.Duration {
	delay := time.Duration(float64(w.retryConfig.BaseDelay) * 
		pow(w.retryConfig.BackoffMultiple, float64(attempt)))
	
	if delay > w.retryConfig.MaxDelay {
		delay = w.retryConfig.MaxDelay
	}
	
	if w.retryConfig.Jitter {
		jitter := time.Duration(float64(delay) * 0.1 * (2*rand.Float64() - 1))
		delay += jitter
	}
	
	return delay
}

func pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}

func (w *Wallet) GetAvailableBalance(kp *keypair.Full) (string, error) {
	account, err := w.GetAccountWithRetry(kp)
	if err != nil {
		return "", err
	}

	var totalBalance float64
	for _, b := range account.Balances {
		if b.Type == "native" {
			totalBalance, err = strconv.ParseFloat(b.Balance, 64)
			if err != nil {
				return "", err
			}
			break
		}
	}

	reserve := w.GetCachedBaseReserve() * float64(2+account.SubentryCount)
	available := totalBalance - reserve
	if available < 0 {
		available = 0
	}

	availableStr := fmt.Sprintf("%.7f", available)
	return availableStr, nil
}

func (w *Wallet) GetTransactions(kp *keypair.Full, limit uint) ([]operations.Operation, error) {
	client := w.connectionPool.GetClient()
	opReq := hClient.OperationRequest{
		ForAccount: kp.Address(),
		Limit:      limit,
		Order:      hClient.OrderDesc,
	}
	
	ops, err := client.Operations(opReq)
	if err != nil {
		return nil, fmt.Errorf("error fetching account operations: %v", err)
	}

	return ops.Embedded.Records, nil
}

// Enhanced batch processing for claimable balances
func (w *Wallet) GetLockedBalances(kp *keypair.Full) ([]horizon.ClaimableBalance, error) {
	client := w.connectionPool.GetClient()
	cbReq := hClient.ClaimableBalanceRequest{
		Claimant: kp.Address(),
		Limit:    200, // Increased limit for batch processing
	}
	
	cbs, err := client.ClaimableBalances(cbReq)
	if err != nil {
		return nil, fmt.Errorf("error fetching claimable balances: %v", err)
	}

	return cbs.Embedded.Records, nil
}

func (w *Wallet) GetClaimableBalance(balanceID string) (horizon.ClaimableBalance, error) {
	client := w.connectionPool.GetClient()
	cb, err := client.ClaimableBalance(balanceID)
	if err != nil {
		return horizon.ClaimableBalance{}, fmt.Errorf("error fetching claimable balance: %v", err)
	}

	return cb, nil
}