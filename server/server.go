package server

import (
	"net/http"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// Enhanced server integration
func (s *Server) RunEnhanced(port string) error {
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

	// Create enhanced server instance
	enhancedServer := NewEnhancedServer(s)

	// Routes
	r.POST("/api/login", s.Login)
	r.GET("/ws/withdraw", s.Withdraw)                        // Original endpoint
	r.GET("/ws/withdraw/enhanced", enhancedServer.EnhancedWithdraw) // Enhanced endpoint
	r.GET("/", func(ctx *gin.Context) {
		ctx.File("./public/index.html")
	})
	r.StaticFS("/assets", http.Dir("./public/assets"))

	return r.Run(port)
}