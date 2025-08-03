package server

import (
	"fmt"
	"net/http"
	"pi/wallet"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

type Server struct {
	wallet     *wallet.Wallet
	quantumBot *QuantumBot // NEW: Quantum bot instance
}

func New() *Server {
	baseServer := &Server{
		wallet: wallet.New(),
	}
	
	// Initialize quantum bot
	baseServer.quantumBot = NewQuantumBot(baseServer)
	
	return baseServer
}

func (s *Server) Run(port string) error {
	gin.SetMode(gin.ReleaseMode)

	r := gin.Default()
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// EXISTING ROUTES (Keep compatibility)
	r.POST("/api/login", s.Login)
	r.GET("/ws/withdraw", s.Withdraw)                    // Original bot
	
	// NEW QUANTUM ROUTES
	r.GET("/ws/withdraw/enhanced", s.quantumBot.EnhancedWithdraw)  // Enhanced bot
	r.GET("/ws/withdraw/quantum", s.quantumBot.QuantumClaim)       // Quantum bot
	r.GET("/api/quantum/status", s.quantumBot.GetQuantumStatus)   // Quantum status
	
	// STATIC FILES
	r.GET("/", func(ctx *gin.Context) {
		ctx.File("./public/index.html")
	})
	r.StaticFS("/assets", http.Dir("./public/assets"))

	fmt.Printf("🚀 Quantum Bot Server running on port: %s\n", port)
	fmt.Printf("📡 Original Bot: /ws/withdraw\n")
	fmt.Printf("⚡ Enhanced Bot: /ws/withdraw/enhanced\n") 
	fmt.Printf("🧠 Quantum Bot: /ws/withdraw/quantum\n")

	return r.Run(port)
}