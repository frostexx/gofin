package server

import (
	"context"
	"fmt"
	"net/http"
	"pi/wallet"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

type Server struct {
	wallet *wallet.Wallet
}

func New() *Server {
	return &Server{
		wallet: wallet.New(),
	}
}

func (s *Server) Run(port string) error {
	gin.SetMode(gin.ReleaseMode)

	r := gin.Default()
	
	// Enhanced CORS configuration
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Authorization", "X-Requested-With"},
		ExposeHeaders:    []string{"Content-Length", "X-Transaction-Hash"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Add request timeout middleware
	r.Use(func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
		defer cancel()
		c.Request = c.Request.WithContext(ctx)
		c.Next()
	})

	r.POST("/api/login", s.Login)
	r.GET("/ws/withdraw", s.Withdraw)
	r.GET("/", func(ctx *gin.Context) {
		ctx.File("./public/index.html")
	})
	r.StaticFS("/assets", http.Dir("./public/assets"))

	fmt.Printf("🚀 High-Performance Pi Bot running on port: %s\n", port)

	return r.Run(port)
}