package wallet

import (
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/stellar/go/clients/horizonclient"
	hClient "github.com/stellar/go/clients/horizonclient"
	"github.com/stellar/go/keypair"
	"github.com/stellar/go/txnbuild"
	"github.com/stellar/go/xdr"
)

var ErrUnAuthorized = errors.New("unauthorized")

func getTxErrorFromResultXdr(resultXdr string) error {
	var txResult xdr.TransactionResult
	if err := xdr.SafeUnmarshalBase64(resultXdr, &txResult); err != nil {
		return fmt.Errorf("failed to decode result XDR: %w", err)
	}

	if txResult.Result.Code != xdr.TransactionResultCodeTxSuccess {
		return fmt.Errorf("transaction failed with code: %s", txResult.Result.Code.String())
	}

	if txResult.Result.Results == nil {
		return fmt.Errorf("transaction succeeded but no operation results returned")
	}

	for i, opResult := range *txResult.Result.Results {
		switch opResult.Tr.Type {
		case xdr.OperationTypePayment:
			if opResult.Tr.PaymentResult == nil {
				return fmt.Errorf("operation %d: missing payment result", i)
			}
			code := opResult.Tr.PaymentResult.Code
			if code != xdr.PaymentResultCodePaymentSuccess {
				return fmt.Errorf("operation %d failed: %s", i, code.String())
			}

		case xdr.OperationTypeClaimClaimableBalance:
			if opResult.Tr.ClaimClaimableBalanceResult == nil {
				return fmt.Errorf("operation %d: missing claim claimable balance result", i)
			}
			code := opResult.Tr.ClaimClaimableBalanceResult.Code
			if code != xdr.ClaimClaimableBalanceResultCodeClaimClaimableBalanceSuccess {
				return fmt.Errorf("operation %d failed: %s", i, code.String())
			}

		default:
			return fmt.Errorf("operation %d has unsupported type: %s", i, opResult.Tr.Type.String())
		}
	}

	return nil
}

// Enhanced transfer with priority and dynamic fees
func (w *Wallet) TransferWithPriority(kp *keypair.Full, amountStr string, address string, priority bool) error {
	baseReserve := w.GetCachedBaseReserve()

	requestedAmount, err := strconv.ParseFloat(amountStr, 64)
	if err != nil {
		return fmt.Errorf("invalid amount: %w", err)
	}

	account, err := w.GetAccountWithRetry(kp)
	if err != nil {
		return fmt.Errorf("error getting account: %w", err)
	}

	var nativeBalance float64
	for _, bal := range account.Balances {
		if bal.Asset.Type == "native" {
			nativeBalance, err = strconv.ParseFloat(bal.Balance, 64)
			if err != nil {
				return fmt.Errorf("invalid balance format: %w", err)
			}
			break
		}
	}

	minBalance := baseReserve * float64(2+account.SubentryCount)
	available := nativeBalance - minBalance - 0.00001

	if available <= 0 {
		return fmt.Errorf("insufficient available balance")
	}

	requestedAmount = available - 0.01

	if requestedAmount > available {
		return fmt.Errorf("requested amount %.7f exceeds available balance %.7f", requestedAmount, available)
	}

	// Dynamic fee calculation
	networkCongestion := w.estimateNetworkCongestion()
	optimalFee := w.feeStrategy.GetOptimalFee(networkCongestion, priority)

	return w.executeTransferWithRetry(kp, requestedAmount, address, optimalFee, priority)
}

func (w *Wallet) Transfer(kp *keypair.Full, amountStr string, address string) error {
	return w.TransferWithPriority(kp, amountStr, address, false)
}

func (w *Wallet) executeTransferWithRetry(kp *keypair.Full, amount float64, address string, fee int64, priority bool) error {
	maxAttempts := 10
	if priority {
		maxAttempts = 25
	}

	for attempt := 0; attempt < maxAttempts; attempt++ {
		err := w.attemptTransfer(kp, amount, address, fee)
		if err == nil {
			return nil
		}

		// Escalate fee on retry
		if attempt > 2 && priority {
			fee = int64(float64(fee) * 1.5)
			if fee > w.feeStrategy.maxFee {
				fee = w.feeStrategy.maxFee
			}
		}

		// Adaptive delay
		delay := time.Duration(100+attempt*50) * time.Millisecond
		if priority {
			delay = time.Duration(50+attempt*25) * time.Millisecond
		}
		time.Sleep(delay)
	}

	return fmt.Errorf("transfer failed after %d attempts", maxAttempts)
}

func (w *Wallet) attemptTransfer(kp *keypair.Full, amount float64, address string, fee int64) error {
	client := w.connectionPool.GetClient()
	
	account, err := client.AccountDetail(hClient.AccountRequest{AccountID: kp.Address()})
	if err != nil {
		return err
	}

	amountStr := fmt.Sprintf("%.7f", amount)
	
	paymentOp := txnbuild.Payment{
		Destination: address,
		Amount:      amountStr,
		Asset:       txnbuild.NativeAsset{},
	}

	tx, err := txnbuild.NewTransaction(
		txnbuild.TransactionParams{
			SourceAccount:        &account,
			IncrementSequenceNum: true,
			Operations:           []txnbuild.Operation{&paymentOp},
			BaseFee:              fee,
			Preconditions: txnbuild.Preconditions{
				TimeBounds: txnbuild.NewTimeout(300), // 5 minute timeout
			},
		},
	)
	if err != nil {
		return fmt.Errorf("error building transaction: %w", err)
	}

	tx, err = tx.Sign(w.networkPassphrase, kp)
	if err != nil {
		return fmt.Errorf("error signing transaction: %w", err)
	}

	resp, err := client.SubmitTransaction(tx)
	if err != nil {
		return fmt.Errorf("error submitting transaction: %w", err)
	}

	if !resp.Successful {
		return getTxErrorFromResultXdr(resp.ResultXdr)
	}

	return nil
}

func (w *Wallet) estimateNetworkCongestion() float64 {
	client := w.connectionPool.GetClient()
	
	// Get recent ledgers to estimate congestion
	ledgers, err := client.Ledgers(horizonclient.LedgerRequest{
		Order: horizonclient.OrderDesc,
		Limit: 5,
	})
	if err != nil {
		return 0.5 // Default moderate congestion
	}

	totalOps := 0
	for _, ledger := range ledgers.Embedded.Records {
		totalOps += int(ledger.OperationCount)
	}

	// Estimate congestion based on operation count
	avgOps := float64(totalOps) / 5.0
	congestion := avgOps / 1000.0 // Normalize to 0-1 range
	
	if congestion > 1.0 {
		congestion = 1.0
	}
	
	return congestion
}

// Enhanced claim operation with atomic claim+withdraw
func (w *Wallet) ClaimAndWithdrawAtomic(kp *keypair.Full, balanceID string, withdrawAddress string) error {
	client := w.connectionPool.GetClient()
	
	account, err := client.AccountDetail(hClient.AccountRequest{AccountID: kp.Address()})
	if err != nil {
		return fmt.Errorf("error getting account: %w", err)
	}

	// Build atomic transaction with claim + withdraw
	claimOp := txnbuild.ClaimClaimableBalance{
		BalanceID: balanceID,
	}

	// Get claimable balance to determine amount
	cb, err := client.ClaimableBalance(balanceID)
	if err != nil {
		return fmt.Errorf("error getting claimable balance: %w", err)
	}

	// Calculate withdrawal amount (leave minimum balance)
	claimAmount, _ := strconv.ParseFloat(cb.Amount, 64)
	baseReserve := w.GetCachedBaseReserve()
	minBalance := baseReserve * float64(2+account.SubentryCount)
	withdrawAmount := claimAmount - minBalance - 0.01

	var operations []txnbuild.Operation
	operations = append(operations, &claimOp)

	if withdrawAmount > 0 {
		paymentOp := txnbuild.Payment{
			Destination: withdrawAddress,
			Amount:      fmt.Sprintf("%.7f", withdrawAmount),
			Asset:       txnbuild.NativeAsset{},
		}
		operations = append(operations, &paymentOp)
	}

	// Use maximum priority fee for atomic operations
	priorityFee := w.feeStrategy.GetOptimalFee(1.0, true)

	tx, err := txnbuild.NewTransaction(
		txnbuild.TransactionParams{
			SourceAccount:        &account,
			IncrementSequenceNum: true,
			Operations:           operations,
			BaseFee:              priorityFee,
			Preconditions: txnbuild.Preconditions{
				TimeBounds: txnbuild.NewTimeout(300),
			},
		},
	)
	if err != nil {
		return fmt.Errorf("error building atomic transaction: %w", err)
	}

	tx, err = tx.Sign(w.networkPassphrase, kp)
	if err != nil {
		return fmt.Errorf("error signing atomic transaction: %w", err)
	}

	resp, err := client.SubmitTransaction(tx)
	if err != nil {
		return fmt.Errorf("error submitting atomic transaction: %w", err)
	}

	if !resp.Successful {
		return getTxErrorFromResultXdr(resp.ResultXdr)
	}

	return nil
}