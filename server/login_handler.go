package server

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// LOGIN HANDLER
type LoginHandler struct {
	bot           *QuantumBot  // Changed from Server to QuantumBot
	sessions      map[string]*UserSession
	loginAttempts map[string]*LoginAttempts
}

type UserSession struct {
	userID    string
	sessionID string
	createdAt time.Time
	expiresAt time.Time
	ipAddress string
	userAgent string
	active    bool
}

type LoginAttempts struct {
	count     int
	lastTry   time.Time
	blocked   bool
	blockTime time.Time
}

type LoginRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type LoginResponse struct {
	Success   bool   `json:"success"`
	Message   string `json:"message"`
	SessionID string `json:"session_id,omitempty"`
	ExpiresAt int64  `json:"expires_at,omitempty"`
}

func NewLoginHandler(bot *QuantumBot) *LoginHandler {  // Changed from Server to QuantumBot
	return &LoginHandler{
		bot:           bot,
		sessions:      make(map[string]*UserSession),
		loginAttempts: make(map[string]*LoginAttempts),
	}
}

func (lh *LoginHandler) HandleLogin(c *gin.Context) {
	var req LoginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, LoginResponse{
			Success: false,
			Message: "Invalid request format",
		})
		return
	}

	clientIP := c.ClientIP()
	
	// Check if IP is blocked due to too many failed attempts
	if lh.isIPBlocked(clientIP) {
		c.JSON(http.StatusTooManyRequests, LoginResponse{
			Success: false,
			Message: "Too many failed login attempts. Please try again later.",
		})
		return
	}

	// Validate credentials (simplified for demo)
	if lh.validateCredentials(req.Username, req.Password) {
		// Reset failed attempts on successful login
		delete(lh.loginAttempts, clientIP)
		
		// Create new session
		session, err := lh.createSession(req.Username, clientIP, c.GetHeader("User-Agent"))
		if err != nil {
			c.JSON(http.StatusInternalServerError, LoginResponse{
				Success: false,
				Message: "Failed to create session",
			})
			return
		}

		c.JSON(http.StatusOK, LoginResponse{
			Success:   true,
			Message:   "Login successful",
			SessionID: session.sessionID,
			ExpiresAt: session.expiresAt.Unix(),
		})
	} else {
		// Record failed attempt
		lh.recordFailedAttempt(clientIP)
		
		c.JSON(http.StatusUnauthorized, LoginResponse{
			Success: false,
			Message: "Invalid username or password",
		})
	}
}

func (lh *LoginHandler) HandleLogout(c *gin.Context) {
	sessionID := c.GetHeader("Authorization")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"message": "No session ID provided",
		})
		return
	}

	// Remove session
	if session, exists := lh.sessions[sessionID]; exists {
		session.active = false
		delete(lh.sessions, sessionID)
		
		c.JSON(http.StatusOK, gin.H{
			"success": true,
			"message": "Logout successful",
		})
	} else {
		c.JSON(http.StatusNotFound, gin.H{
			"success": false,
			"message": "Session not found",
		})
	}
}

func (lh *LoginHandler) HandleSessionCheck(c *gin.Context) {
	sessionID := c.GetHeader("Authorization")
	if sessionID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{
			"valid": false,
			"message": "No session ID provided",
		})
		return
	}

	session, valid := lh.isSessionValid(sessionID)
	if valid {
		c.JSON(http.StatusOK, gin.H{
			"valid":      true,
			"user_id":    session.userID,
			"expires_at": session.expiresAt.Unix(),
		})
	} else {
		c.JSON(http.StatusUnauthorized, gin.H{
			"valid":   false,
			"message": "Invalid or expired session",
		})
	}
}

func (lh *LoginHandler) validateCredentials(username, password string) bool {
	// Simplified credential validation
	// In a real application, this would check against a secure database
	validCredentials := map[string]string{
		"admin":    "quantum2024",
		"operator": "stellar123",
		"viewer":   "readonly",
	}

	if validPassword, exists := validCredentials[username]; exists {
		return password == validPassword
	}
	return false
}

func (lh *LoginHandler) createSession(userID, ipAddress, userAgent string) (*UserSession, error) {
	// Generate secure session ID
	sessionBytes := make([]byte, 32)
	_, err := rand.Read(sessionBytes)
	if err != nil {
		return nil, err
	}
	sessionID := hex.EncodeToString(sessionBytes)

	session := &UserSession{
		userID:    userID,
		sessionID: sessionID,
		createdAt: time.Now(),
		expiresAt: time.Now().Add(24 * time.Hour), // 24 hour session
		ipAddress: ipAddress,
		userAgent: userAgent,
		active:    true,
	}

	lh.sessions[sessionID] = session
	return session, nil
}

func (lh *LoginHandler) isSessionValid(sessionID string) (*UserSession, bool) {
	session, exists := lh.sessions[sessionID]
	if !exists {
		return nil, false
	}

	if !session.active || time.Now().After(session.expiresAt) {
		// Clean up expired session
		delete(lh.sessions, sessionID)
		return nil, false
	}

	return session, true
}

func (lh *LoginHandler) isIPBlocked(ip string) bool {
	attempts, exists := lh.loginAttempts[ip]
	if !exists {
		return false
	}

	if attempts.blocked {
		// Check if block time has expired (15 minutes)
		if time.Since(attempts.blockTime) > 15*time.Minute {
			attempts.blocked = false
			attempts.count = 0
			return false
		}
		return true
	}

	return false
}

func (lh *LoginHandler) recordFailedAttempt(ip string) {
	attempts, exists := lh.loginAttempts[ip]
	if !exists {
		attempts = &LoginAttempts{
			count:   0,
			lastTry: time.Now(),
		}
		lh.loginAttempts[ip] = attempts
	}

	attempts.count++
	attempts.lastTry = time.Now()

	// Block IP after 5 failed attempts
	if attempts.count >= 5 {
		attempts.blocked = true
		attempts.blockTime = time.Now()
		
		fmt.Printf("🔒 IP %s blocked due to too many failed login attempts\n", ip)
	}
}

func (lh *LoginHandler) AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		sessionID := c.GetHeader("Authorization")
		if sessionID == "" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "No authorization header provided",
			})
			c.Abort()
			return
		}

		session, valid := lh.isSessionValid(sessionID)
		if !valid {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "Invalid or expired session",
			})
			c.Abort()
			return
		}

		// Store user info in context
		c.Set("user_id", session.userID)
		c.Set("session_id", session.sessionID)
		
		c.Next()
	}
}

func (lh *LoginHandler) CleanupExpiredSessions() {
	now := time.Now()
	
	for sessionID, session := range lh.sessions {
		if now.After(session.expiresAt) || !session.active {
			delete(lh.sessions, sessionID)
		}
	}
	
	// Clean up old login attempts (older than 1 hour)
	for ip, attempts := range lh.loginAttempts {
		if now.Sub(attempts.lastTry) > 1*time.Hour {
			delete(lh.loginAttempts, ip)
		}
	}
}

func (lh *LoginHandler) GetActiveSessions() map[string]interface{} {
	activeSessions := make([]map[string]interface{}, 0)
	
	for _, session := range lh.sessions {
		if session.active && time.Now().Before(session.expiresAt) {
			activeSessions = append(activeSessions, map[string]interface{}{
				"user_id":    session.userID,
				"created_at": session.createdAt.Unix(),
				"expires_at": session.expiresAt.Unix(),
				"ip_address": session.ipAddress,
			})
		}
	}
	
	return map[string]interface{}{
		"active_sessions": activeSessions,
		"total_count":     len(activeSessions),
	}
}

// ADMIN FUNCTIONS
func (lh *LoginHandler) ForceLogoutUser(userID string) int {
	loggedOut := 0
	
	for sessionID, session := range lh.sessions {
		if session.userID == userID && session.active {
			session.active = false
			delete(lh.sessions, sessionID)
			loggedOut++
		}
	}
	
	return loggedOut
}

func (lh *LoginHandler) UnblockIP(ip string) bool {
	if attempts, exists := lh.loginAttempts[ip]; exists {
		attempts.blocked = false
		attempts.count = 0
		return true
	}
	return false
}

func (lh *LoginHandler) GetBlockedIPs() []string {
	blocked := make([]string, 0)
	
	for ip, attempts := range lh.loginAttempts {
		if attempts.blocked {
			blocked = append(blocked, ip)
		}
	}
	
	return blocked
}