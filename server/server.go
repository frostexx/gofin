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
	wallet         *wallet.Wallet
	enhancedWallet *wallet.EnhancedWallet // NEW: Enhanced wallet instance
}

func New() *Server {
	return &Server{
		wallet:         wallet.New(),
		enhancedWallet: wallet.NewEnhancedWallet(), // NEW: Initialize enhanced wallet
	}
}

func (s *Server) Run(port string) error {
	gin.SetMode(gin.ReleaseMode)

	r := gin.Default()
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"}, // or restrict to specific domains
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Legacy routes (for backward compatibility)
	r.POST("/api/login", s.Login)
	r.GET("/ws/withdraw", s.Withdraw)
	
	// NEW: Enhanced routes for superior performance
	r.GET("/ws/enhanced-withdraw", s.EnhancedWithdraw)
	r.GET("/ws/turbo-withdraw", s.TurboWithdraw)
	r.GET("/api/performance-metrics", s.GetPerformanceMetrics)
	r.POST("/api/configure-strategy", s.ConfigureStrategy)
	
	// Static files
	r.GET("/", func(ctx *gin.Context) {
		ctx.File("./public/index.html")
	})
	r.StaticFS("/assets", http.Dir("./public/assets"))

	fmt.Printf("running on port: %s\n", port)

	return r.Run(port)
}