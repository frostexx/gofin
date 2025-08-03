package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"pi/server"
	
	"github.com/fatih/color"
	"github.com/joho/godotenv"
)

func main() {
	// Load environment variables
	err := godotenv.Load()
	if err != nil {
		log.Println("No .env file found, using system environment variables")
	}

	// Print startup banner
	printStartupBanner()

	// Get port from environment
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Create and configure QuantumBot
	quantumBot := server.NewQuantumBot()

	// Setup graceful shutdown
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		fmt.Println("\n🛑 Shutting down QuantumBot...")
		quantumBot.Stop()
		os.Exit(0)
	}()

	// Start the server
	fmt.Printf("🌟 Starting QuantumBot on port %s...\n", port)
	
	if err := quantumBot.Start(port); err != nil {
		log.Fatal("❌ Failed to start server:", err)
	}
}

func printStartupBanner() {
	cyan := color.New(color.FgCyan, color.Bold)
	magenta := color.New(color.FgMagenta, color.Bold)
	yellow := color.New(color.FgYellow, color.Bold)
	green := color.New(color.FgGreen, color.Bold)

	fmt.Println()
	cyan.Println("  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗███╗   ███╗")
	cyan.Println(" ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║████╗ ████║")
	cyan.Println(" ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║")
	cyan.Println(" ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║")
	cyan.Println(" ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║")
	cyan.Println("  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝")
	fmt.Println()
	
	magenta.Println("  ██████╗  ██████╗ ████████╗")
	magenta.Println("  ██╔══██╗██╔═══██╗╚══██╔══╝")
	magenta.Println("  ██████╔╝██║   ██║   ██║   ")
	magenta.Println("  ██╔══██╗██║   ██║   ██║   ")
	magenta.Println("  ██████╔╝╚██████╔╝   ██║   ")
	magenta.Println("  ╚═════╝  ╚═════╝    ╚═╝   ")
	fmt.Println()

	yellow.Println("🚀 QUANTUM-ENHANCED STELLAR AUTOMATION SYSTEM")
	fmt.Println()
	
	green.Println("⚛️  Quantum Integration Systems")
	green.Println("🧠 AI Learning & Prediction Engine")
	green.Println("🔥 Hardware-Level Optimizations")
	green.Println("🌐 Distributed Swarm Intelligence")
	green.Println("⚔️  Advanced Network Warfare (Educational)")
	green.Println("🔒 Post-Quantum Cryptography")
	green.Println("📊 Real-Time Performance Analytics")
	green.Println("🎯 Precision Timing & Execution")
	
	fmt.Println()
	cyan.Printf("💫 System Status: ")
	green.Println("INITIALIZING...")
	fmt.Println()
}