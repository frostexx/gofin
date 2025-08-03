package server

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stellar/go/clients/horizonclient"
	"github.com/stellar/go/network"
	"github.com/stellar/go/protocols/horizon"
	"github.com/stellar/go/txnbuild"
)

// TRANSACTION HANDLER
type TransactionHandler struct {
	bot            *QuantumBot  // Changed from Server to QuantumBot
	stellarClient  *horizonclient.Client
	networkMonitor *TransactionNetworkMonitor  // Renamed to avoid conflict
}

type TransactionNetworkMonitor struct {
	transactions    []StellarTransaction
	successRate     float64
	averageLatency  time.Duration
	lastUpdate      time.Time
	feeTrends       []FeeTrend
}

type StellarTransaction struct {
	Hash           string
	SourceAccount  string
	Sequence       int64
	Fee            int64
	OperationCount int
	Timestamp      time.Time
	Status         string
	Ledger         int32
}

type FeeTrend struct {
	Timestamp   time.Time
	AverageFee  int64
	MedianFee   int64
	MinFee      int64
	MaxFee      int64
	TxCount     int
}

func NewTransactionHandler(bot *QuantumBot) *TransactionHandler {  // Changed from Server to QuantumBot
	client := horizonclient.DefaultTestNetClient
	
	return &TransactionHandler{
		bot:           bot,
		stellarClient: client,
		networkMonitor: &TransactionNetworkMonitor{
			transactions: make([]StellarTransaction, 0),
			feeTrends:    make([]FeeTrend, 0),
			lastUpdate:   time.Now(),
		},
	}
}

func (th *TransactionHandler) MonitorTransactions(c *gin.Context) {
	// Update network monitor
	th.updateNetworkMetrics()
	
	c.JSON(http.StatusOK, gin.H{
		"status":        "monitoring",
		"success_rate":  th.networkMonitor.successRate,
		"avg_latency":   th.networkMonitor.averageLatency.Milliseconds(),
		"last_update":   th.networkMonitor.lastUpdate,
		"transaction_count": len(th.networkMonitor.transactions),
	})
}

func (th *TransactionHandler) GetTransactionHistory(c *gin.Context) {
	// Get recent transactions
	recentTxs := th.getRecentTransactions(100)
	
	c.JSON(http.StatusOK, gin.H{
		"transactions": recentTxs,
		"count":       len(recentTxs),
		"timestamp":   time.Now(),
	})
}

func (th *TransactionHandler) OptimizeTransaction(c *gin.Context) {
	var req struct {
		SourceAccount string `json:"source_account"`
		Amount        string `json:"amount"`
		Destination   string `json:"destination"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Use quantum systems for optimization
	optimalFee := th.calculateOptimalFee()
	optimalTiming := th.bot.getQuantumPrecisionTime()
	
	c.JSON(http.StatusOK, gin.H{
		"optimal_fee":    optimalFee,
		"optimal_timing": optimalTiming.Unix(),
		"confidence":     th.calculateConfidence(),
		"quantum_enhanced": true,
	})
}

func (th *TransactionHandler) updateNetworkMetrics() {
	// Fetch recent transactions from Stellar network
	txs, err := th.fetchRecentTransactions()
	if err != nil {
		return
	}
	
	// Update success rate
	successCount := 0
	totalLatency := time.Duration(0)
	
	for _, tx := range txs {
		if tx.Status == "success" {
			successCount++
		}
		// Calculate latency (simplified)
		latency := time.Since(tx.Timestamp)
		totalLatency += latency
	}
	
	if len(txs) > 0 {
		th.networkMonitor.successRate = float64(successCount) / float64(len(txs))
		th.networkMonitor.averageLatency = totalLatency / time.Duration(len(txs))
	}
	
	th.networkMonitor.transactions = txs
	th.networkMonitor.lastUpdate = time.Now()
	
	// Update fee trends
	th.updateFeeTrends(txs)
}

func (th *TransactionHandler) fetchRecentTransactions() ([]StellarTransaction, error) {
	// Fetch transactions from Stellar Horizon API
	request := horizonclient.TransactionRequest{
		Limit: 200,
		Order: horizonclient.OrderDesc,
	}
	
	txPage, err := th.stellarClient.Transactions(request)
	if err != nil {
		return nil, err
	}
	
	transactions := make([]StellarTransaction, 0)
	
	for _, tx := range txPage.Embedded.Records {
		stellarTx := StellarTransaction{
			Hash:          tx.Hash,
			SourceAccount: tx.SourceAccount,
			Sequence:      tx.SourceAccountSequence,
			Fee:           tx.FeePaid,
			OperationCount: tx.OperationCount,
			Timestamp:     tx.CreatedAt,
			Status:        determineStatus(tx),
			Ledger:        tx.Ledger,
		}
		
		transactions = append(transactions, stellarTx)
	}
	
	return transactions, nil
}

func determineStatus(tx horizon.Transaction) string {
	if tx.Successful {
		return "success"
	}
	return "failed"
}

func (th *TransactionHandler) updateFeeTrends(transactions []StellarTransaction) {
	if len(transactions) == 0 {
		return
	}
	
	// Group transactions by time periods (e.g., every 5 minutes)
	feeTrend := FeeTrend{
		Timestamp: time.Now(),
		TxCount:   len(transactions),
	}
	
	totalFee := int64(0)
	minFee := transactions[0].Fee
	maxFee := transactions[0].Fee
	
	for _, tx := range transactions {
		totalFee += tx.Fee
		if tx.Fee < minFee {
			minFee = tx.Fee
		}
		if tx.Fee > maxFee {
			maxFee = tx.Fee
		}
	}
	
	feeTrend.AverageFee = totalFee / int64(len(transactions))
	feeTrend.MinFee = minFee
	feeTrend.MaxFee = maxFee
	
	// Calculate median (simplified)
	feeTrend.MedianFee = feeTrend.AverageFee
	
	th.networkMonitor.feeTrends = append(th.networkMonitor.feeTrends, feeTrend)
	
	// Keep only recent trends (last 24 hours)
	cutoff := time.Now().Add(-24 * time.Hour)
	filteredTrends := make([]FeeTrend, 0)
	for _, trend := range th.networkMonitor.feeTrends {
		if trend.Timestamp.After(cutoff) {
			filteredTrends = append(filteredTrends, trend)
		}
	}
	th.networkMonitor.feeTrends = filteredTrends
}

func (th *TransactionHandler) calculateOptimalFee() int64 {
	if len(th.networkMonitor.feeTrends) == 0 {
		return 100 // Default fee
	}
	
	// Use recent trends to calculate optimal fee
	recentTrend := th.networkMonitor.feeTrends[len(th.networkMonitor.feeTrends)-1]
	
	// Add 10% buffer to average fee for faster confirmation
	optimalFee := int64(float64(recentTrend.AverageFee) * 1.1)
	
	return optimalFee
}

func (th *TransactionHandler) calculateConfidence() float64 {
	// Calculate confidence based on network stability and AI predictions
	if th.bot.neuralPredictor == nil {
		return 0.7 // Default confidence
	}
	
	// Combine network metrics with AI confidence
	networkStability := th.networkMonitor.successRate
	aiConfidence := th.bot.performanceMetrics.AIAccuracy
	
	return (networkStability + aiConfidence) / 2.0
}

func (th *TransactionHandler) getRecentTransactions(limit int) []StellarTransaction {
	if len(th.networkMonitor.transactions) <= limit {
		return th.networkMonitor.transactions
	}
	
	return th.networkMonitor.transactions[:limit]
}

// TRANSACTION BUILDING HELPERS
func (th *TransactionHandler) BuildOptimizedTransaction(sourceKP *txnbuild.SimpleKeypair, destAccount string, amount string) (*txnbuild.Transaction, error) {
	// Get source account details
	sourceAccountRequest := horizonclient.AccountRequest{AccountID: sourceKP.Address()}
	sourceAccount, err := th.stellarClient.AccountDetail(sourceAccountRequest)
	if err != nil {
		return nil, err
	}
	
	// Calculate optimal fee using quantum systems
	optimalFee := th.calculateOptimalFee()
	
	// Build payment operation
	paymentOp := txnbuild.Payment{
		Destination: destAccount,
		Amount:      amount,
		Asset:       txnbuild.NativeAsset{},
	}
	
	// Build transaction with quantum-optimized fee
	tx, err := txnbuild.NewTransaction(
		txnbuild.TransactionParams{
			SourceAccount:        &sourceAccount,
			IncrementSequenceNum: true,
			Operations:           []txnbuild.Operation{&paymentOp},
			BaseFee:              optimalFee,
			Memo:                 txnbuild.MemoText("Quantum Optimized"),
			Preconditions: txnbuild.Preconditions{
				TimeBounds: txnbuild.NewInfiniteTimeout(),
			},
		},
	)
	
	if err != nil {
		return nil, err
	}
	
	return tx, nil
}

func (th *TransactionHandler) SubmitWithQuantumTiming(tx *txnbuild.Transaction, signers ...*txnbuild.SimpleKeypair) (*horizon.Transaction, error) {
	// Sign transaction
	for _, signer := range signers {
		tx, err := tx.Sign(network.TestNetworkPassphrase, signer)
		if err != nil {
			return nil, err
		}
	}
	
	// Wait for optimal quantum timing
	optimalTime := th.bot.getQuantumPrecisionTime()
	waitDuration := time.Until(optimalTime)
	
	if waitDuration > 0 && waitDuration < 5*time.Second {
		time.Sleep(waitDuration)
	}
	
	// Submit transaction
	resp, err := th.stellarClient.SubmitTransaction(tx)
	if err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// TRANSACTION ANALYSIS
func (th *TransactionHandler) AnalyzeNetworkConditions() map[string]interface{} {
	return map[string]interface{}{
		"success_rate":    th.networkMonitor.successRate,
		"avg_latency_ms":  th.networkMonitor.averageLatency.Milliseconds(),
		"transaction_count": len(th.networkMonitor.transactions),
		"fee_trends":      th.getFeeTrendSummary(),
		"network_health":  th.calculateNetworkHealth(),
		"optimal_fee":     th.calculateOptimalFee(),
		"quantum_timing":  th.bot.getQuantumPrecisionTime().Unix(),
	}
}

func (th *TransactionHandler) getFeeTrendSummary() map[string]interface{} {
	if len(th.networkMonitor.feeTrends) == 0 {
		return map[string]interface{}{
			"trend": "no_data",
		}
	}
	
	recent := th.networkMonitor.feeTrends[len(th.networkMonitor.feeTrends)-1]
	
	return map[string]interface{}{
		"average_fee": recent.AverageFee,
		"min_fee":     recent.MinFee,
		"max_fee":     recent.MaxFee,
		"tx_count":    recent.TxCount,
		"timestamp":   recent.Timestamp,
	}
}

func (th *TransactionHandler) calculateNetworkHealth() string {
	successRate := th.networkMonitor.successRate
	avgLatency := th.networkMonitor.averageLatency
	
	if successRate > 0.95 && avgLatency < 5*time.Second {
		return "excellent"
	} else if successRate > 0.90 && avgLatency < 10*time.Second {
		return "good"
	} else if successRate > 0.80 && avgLatency < 20*time.Second {
		return "fair"
	} else {
		return "poor"
	}
}