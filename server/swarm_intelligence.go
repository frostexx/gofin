package server

import (
	"crypto/sha256"
	"fmt"
	"math"
	"sync"
	"time"
)

// SWARM INTELLIGENCE TYPES (removed duplicates)
type SwarmNode struct {
	ID                string
	Address           string
	PublicKey         []byte
	Reputation        float64
	LastSeen          time.Time
	Active            bool
	Performance       NodePerformance
	Capabilities      []string
}

type NodePerformance struct {
	Latency       time.Duration
	Throughput    float64
	Uptime        float64
	ErrorRate     float64
	LastUpdate    time.Time
}

type ConsensusEngine struct {
	algorithm       ConsensusAlgorithm
	participants    []*SwarmNode
	currentRound    int64
	votingThreshold float64
	lastConsensus   time.Time
	consensusValue  interface{}
	mutex           sync.RWMutex
}

type ConsensusAlgorithm int

const (
	AlgorithmPBFT ConsensusAlgorithm = iota
	AlgorithmRaft
	AlgorithmDPoS
	AlgorithmHoneyBadger
)

type DistributedLedger struct {
	blocks          []Block
	pendingTxs      []DistributedTransaction
	validators      []*SwarmNode
	consensusEngine *ConsensusEngine
	mutex           sync.RWMutex
}

type Block struct {
	Index        int64
	Timestamp    time.Time
	PreviousHash []byte
	Hash         []byte
	Transactions []DistributedTransaction
	Validator    string
	Signature    []byte
}

type DistributedTransaction struct {
	ID        string
	Type      string
	Payload   []byte
	Timestamp time.Time
	Signature []byte
	Valid     bool
}

// SWARM INTELLIGENCE INITIALIZATION
func (qb *QuantumBot) initializeSwarmIntelligence() {
	fmt.Println("🧠 Initializing Swarm Intelligence...")
	
	// Initialize swarm nodes
	qb.initializeSwarmNodes()
	
	// Initialize consensus engine
	qb.initializeConsensusEngine()
	
	// Initialize distributed ledger
	qb.initializeDistributedLedger()
	
	// Start swarm workers
	go qb.swarmCoordinationWorker()
	go qb.consensusWorker()
	go qb.ledgerWorker()
	
	fmt.Println("🧠 Swarm Intelligence initialized successfully!")
}

func (qb *QuantumBot) initializeSwarmNodes() {
	// Create initial swarm nodes
	nodeCount := 5
	qb.swarmNodes = make([]*SwarmNode, nodeCount)
	
	for i := 0; i < nodeCount; i++ {
		node := &SwarmNode{
			ID:        fmt.Sprintf("node_%d", i),
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Reputation: 0.8 + (float64(i)*0.05), // Varying reputation
			LastSeen:  time.Now(),
			Active:    true,
			Performance: NodePerformance{
				Latency:    time.Duration(i*10) * time.Millisecond,
				Throughput: 1000.0 - float64(i*50),
				Uptime:     0.99 - float64(i)*0.01,
				ErrorRate:  float64(i) * 0.001,
				LastUpdate: time.Now(),
			},
			Capabilities: []string{"consensus", "validation", "storage"},
		}
		
		// Generate public key (simplified)
		node.PublicKey = make([]byte, 32)
		copy(node.PublicKey, qb.getQuantumRandom())
		
		qb.swarmNodes[i] = node
	}
}

func (qb *QuantumBot) initializeConsensusEngine() {
	qb.consensusEngine = &ConsensusEngine{
		algorithm:       AlgorithmPBFT,
		participants:    qb.swarmNodes,
		currentRound:    0,
		votingThreshold: 0.67, // 2/3 majority
		lastConsensus:   time.Now(),
	}
}

func (qb *QuantumBot) initializeDistributedLedger() {
	qb.distributedLedger = &DistributedLedger{
		blocks:          make([]Block, 0),
		pendingTxs:      make([]DistributedTransaction, 0),
		validators:      qb.swarmNodes,
		consensusEngine: qb.consensusEngine,
	}
	
	// Create genesis block
	genesisBlock := Block{
		Index:        0,
		Timestamp:    time.Now(),
		PreviousHash: make([]byte, 32),
		Transactions: make([]DistributedTransaction, 0),
		Validator:    "genesis",
	}
	
	genesisBlock.Hash = qb.calculateBlockHash(genesisBlock)
	qb.distributedLedger.blocks = append(qb.distributedLedger.blocks, genesisBlock)
}

// SWARM WORKERS
func (qb *QuantumBot) swarmCoordinationWorker() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		qb.coordinateSwarmNodes()
	}
}

func (qb *QuantumBot) coordinateSwarmNodes() {
	// Update node statuses
	for _, node := range qb.swarmNodes {
		qb.updateNodeStatus(node)
	}
	
	// Optimize swarm configuration
	qb.optimizeSwarmConfiguration()
}

func (qb *QuantumBot) updateNodeStatus(node *SwarmNode) {
	// Simulate node status update
	node.LastSeen = time.Now()
	
	// Update performance metrics
	node.Performance.LastUpdate = time.Now()
	
	// Adjust reputation based on performance
	if node.Performance.ErrorRate < 0.01 {
		node.Reputation = math.Min(1.0, node.Reputation+0.01)
	} else {
		node.Reputation = math.Max(0.0, node.Reputation-0.02)
	}
}

func (qb *QuantumBot) optimizeSwarmConfiguration() {
	// Remove inactive nodes
	activeNodes := make([]*SwarmNode, 0)
	for _, node := range qb.swarmNodes {
		if time.Since(node.LastSeen) < 30*time.Second {
			activeNodes = append(activeNodes, node)
		}
	}
	qb.swarmNodes = activeNodes
	
	// Update consensus participants
	qb.consensusEngine.participants = activeNodes
}

func (qb *QuantumBot) consensusWorker() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		qb.performConsensus()
	}
}

func (qb *QuantumBot) performConsensus() {
	if len(qb.consensusEngine.participants) < 3 {
		return // Need minimum nodes for consensus
	}
	
	// Simulate consensus round
	qb.consensusEngine.currentRound++
	
	// Calculate consensus value based on node votes
	consensusReached := qb.calculateConsensus()
	
	if consensusReached {
		qb.consensusEngine.lastConsensus = time.Now()
		fmt.Printf("🧠 Consensus achieved in round %d\n", qb.consensusEngine.currentRound)
	}
}

func (qb *QuantumBot) calculateConsensus() bool {
	// Simplified consensus calculation
	participantCount := len(qb.consensusEngine.participants)
	if participantCount == 0 {
		return false
	}
	
	// Count active participants
	activeCount := 0
	for _, node := range qb.consensusEngine.participants {
		if node.Active && node.Reputation > 0.5 {
			activeCount++
		}
	}
	
	// Check if we have enough participants
	threshold := int(float64(participantCount) * qb.consensusEngine.votingThreshold)
	return activeCount >= threshold
}

func (qb *QuantumBot) ledgerWorker() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		qb.processLedgerTransactions()
	}
}

func (qb *QuantumBot) processLedgerTransactions() {
	qb.distributedLedger.mutex.Lock()
	defer qb.distributedLedger.mutex.Unlock()
	
	if len(qb.distributedLedger.pendingTxs) == 0 {
		return
	}
	
	// Create new block with pending transactions
	newBlock := qb.createNewBlock()
	
	// Validate block
	if qb.validateBlock(newBlock) {
		qb.distributedLedger.blocks = append(qb.distributedLedger.blocks, newBlock)
		qb.distributedLedger.pendingTxs = make([]DistributedTransaction, 0)
		
		fmt.Printf("📊 New block added to ledger: #%d\n", newBlock.Index)
	}
}

func (qb *QuantumBot) createNewBlock() Block {
	prevBlock := qb.distributedLedger.blocks[len(qb.distributedLedger.blocks)-1]
	
	newBlock := Block{
		Index:        prevBlock.Index + 1,
		Timestamp:    time.Now(),
		PreviousHash: prevBlock.Hash,
		Transactions: make([]DistributedTransaction, len(qb.distributedLedger.pendingTxs)),
		Validator:    qb.selectValidator(),
	}
	
	copy(newBlock.Transactions, qb.distributedLedger.pendingTxs)
	newBlock.Hash = qb.calculateBlockHash(newBlock)
	
	return newBlock
}

func (qb *QuantumBot) selectValidator() string {
	// Select validator based on reputation
	bestNode := qb.swarmNodes[0]
	for _, node := range qb.swarmNodes[1:] {
		if node.Reputation > bestNode.Reputation && node.Active {
			bestNode = node
		}
	}
	
	return bestNode.ID
}

func (qb *QuantumBot) calculateBlockHash(block Block) []byte {
	// Simple hash calculation
	hasher := sha256.New()
	hasher.Write([]byte(fmt.Sprintf("%d%d%x", block.Index, block.Timestamp.Unix(), block.PreviousHash)))
	
	for _, tx := range block.Transactions {
		hasher.Write([]byte(tx.ID))
	}
	
	return hasher.Sum(nil)
}

func (qb *QuantumBot) validateBlock(block Block) bool {
	// Simple block validation
	if len(qb.distributedLedger.blocks) == 0 {
		return true // Genesis block
	}
	
	prevBlock := qb.distributedLedger.blocks[len(qb.distributedLedger.blocks)-1]
	
	// Check index
	if block.Index != prevBlock.Index+1 {
		return false
	}
	
	// Check previous hash
	for i, b := range block.PreviousHash {
		if b != prevBlock.Hash[i] {
			return false
		}
	}
	
	// Check timestamp
	if block.Timestamp.Before(prevBlock.Timestamp) {
		return false
	}
	
	return true
}

// CONSENSUS METHODS
func (qb *QuantumBot) getCurrentConsensus() float64 {
	if qb.consensusEngine == nil {
		return 0.0
	}
	
	qb.consensusEngine.mutex.RLock()
	defer qb.consensusEngine.mutex.RUnlock()
	
	// Calculate consensus strength based on active participants
	activeCount := 0
	totalReputation := 0.0
	
	for _, node := range qb.consensusEngine.participants {
		if node.Active {
			activeCount++
			totalReputation += node.Reputation
		}
	}
	
	if activeCount == 0 {
		return 0.0
	}
	
	averageReputation := totalReputation / float64(activeCount)
	participationRate := float64(activeCount) / float64(len(qb.consensusEngine.participants))
	
	return averageReputation * participationRate
}

func (qb *QuantumBot) addSwarmTransaction(txType string, payload []byte) {
	tx := DistributedTransaction{
		ID:        fmt.Sprintf("tx_%d_%s", time.Now().Unix(), txType),
		Type:      txType,
		Payload:   payload,
		Timestamp: time.Now(),
		Valid:     true,
	}
	
	// Sign transaction (simplified)
	hasher := sha256.New()
	hasher.Write([]byte(tx.ID))
	hasher.Write(payload)
	tx.Signature = hasher.Sum(nil)
	
	qb.distributedLedger.mutex.Lock()
	qb.distributedLedger.pendingTxs = append(qb.distributedLedger.pendingTxs, tx)
	qb.distributedLedger.mutex.Unlock()
}