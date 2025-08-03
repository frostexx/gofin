package wallet

import (
	"fmt"
	"strconv"
	"sync"
	
	"github.com/stellar/go/clients/horizonclient"
	hClient "github.com/stellar/go/clients/horizonclient"
	"github.com/stellar/go/keypair"
	"github.com/stellar/go/protocols/horizon"
	"github.com/stellar/go/txnbuild"
)

// Base Wallet struct definition
type Wallet struct {
    client            *horizonclient.Client
    networkPassphrase string
    baseReserve       float64
    mutex             sync.RWMutex
}

// Constructor for base wallet
func New() *Wallet {
    return &Wallet{
        client:            horizonclient.DefaultTestNetClient,
        networkPassphrase: "Test SDF Network ; September 2015",
        baseReserve:       0.5,
    }
}

// GetAccount fetches the details of a Stellar account.
func (w *Wallet) GetAccount(kp *keypair.Full) (horizon.Account, error) {
	accReq := hClient.AccountRequest{AccountID: kp.Address()}
	account, err := w.client.AccountDetail(accReq)
	if err != nil {
		return horizon.Account{}, fmt.Errorf("error fetching account details: %v", err)
	}
	return account, nil
}

// GetAvailableBalance calculates the spendable balance of an account.
func (w *Wallet) GetAvailableBalance(kp *keypair.Full) (string, error) {
	account, err := w.GetAccount(kp)
	if err != nil {
		return "", err
	}

	var totalBalance float64
	for _, b := range account.Balances {
		if b.Type == "native" {
			totalBalance, err = strconv.ParseFloat(b.Balance, 64)
			if err != nil {
				return "", err
			}
			break
		}
	}

	reserve := w.baseReserve * float64(2+account.SubentryCount)
	available := totalBalance - reserve
	if available < 0 {
		available = 0
	}

	availableStr := fmt.Sprintf("%.2f", available)
	return availableStr, nil
}

// GetLockedBalances is a placeholder for fetching locked balances (e.g., claimable balances).
func (w *Wallet) GetLockedBalances(kp *keypair.Full) ([]LockedBalance, error) {
    // Implementation needed for getting locked balances
    return []LockedBalance{}, nil
}

// GetTransactions is a placeholder for fetching an account's transaction history.
func (w *Wallet) GetTransactions(kp *keypair.Full, limit int) ([]Transaction, error) {
    // Implementation needed for getting transactions
    return []Transaction{}, nil
}

// Login parses a seed phrase to get a keypair.
func (w *Wallet) Login(seedPhrase string) (*keypair.Full, error) {
	kp, err := keypair.ParseFull(seedPhrase)
	if err != nil {
		return nil, fmt.Errorf("invalid seed phrase: %v", err)
	}
	return kp, nil
}

// ClaimAndWithdraw is a placeholder for a simple claim and withdraw operation.
func (w *Wallet) ClaimAndWithdraw(kp *keypair.Full, amount float64, balanceID, address string) (string, error) {
    // Implementation for claim and withdraw
    return "", nil
}

// GetBaseReserve is a placeholder for fetching the current base reserve from the network.
func (w *Wallet) GetBaseReserve() {
    // In a real app, this would fetch the latest base reserve from the network's ledger data.
    w.baseReserve = 0.5
}

// TransferWithDynamicFee sends a payment using a specified dynamic fee.
func (w *Wallet) TransferWithDynamicFee(kp *keypair.Full, amountStr string, address string, dynamicFee float64) error {
	account, err := w.GetAccount(kp)
	if err != nil {
		return fmt.Errorf("error getting account: %w", err)
	}

	paymentOp := txnbuild.Payment{
		Destination: address,
		Amount:      amountStr,
		Asset:       txnbuild.NativeAsset{},
	}

	txParams := txnbuild.TransactionParams{
		SourceAccount:        &account,
		IncrementSequenceNum: true,
		Operations:           []txnbuild.Operation{&paymentOp},
		BaseFee:              int64(dynamicFee),
		Preconditions:        txnbuild.Preconditions{TimeBounds: txnbuild.NewInfiniteTimeout()},
	}

	tx, err := txnbuild.NewTransaction(txParams)
	if err != nil {
		return fmt.Errorf("error building transaction: %w", err)
	}

	signedTx, err := tx.Sign(w.networkPassphrase, kp)
	if err != nil {
		return fmt.Errorf("error signing transaction: %w", err)
	}

	resp, err := w.client.SubmitTransaction(signedTx)
	if err != nil {
		return fmt.Errorf("error submitting transaction: %w", err)
	}

	if !resp.Successful {
		return fmt.Errorf("transaction failed: %s", resp.ResultXdr)
	}

	return nil
}

// WithdrawClaimableBalanceWithDynamicFee claims a balance and sends it to a destination using a dynamic fee.
func (w *Wallet) WithdrawClaimableBalanceWithDynamicFee(kp *keypair.Full, amount, balanceID, address string, dynamicFee float64) (string, float64, error) {
	account, err := w.GetAccount(kp)
	if err != nil {
		return "", 0, err
	}

	amountFloat, err := strconv.ParseFloat(amount, 64)
	if err != nil {
		return "", 0, fmt.Errorf("invalid amount: %v", err)
	}

	claimOp := txnbuild.ClaimClaimableBalance{
		BalanceID: balanceID,
	}

	paymentOp := txnbuild.Payment{
		Destination: address,
		Amount:      amount,
		Asset:       txnbuild.NativeAsset{},
	}

	txParams := txnbuild.TransactionParams{
		SourceAccount:        &account,
		IncrementSequenceNum: true,
		Operations:           []txnbuild.Operation{&claimOp, &paymentOp},
		BaseFee:              int64(dynamicFee),
		Preconditions: txnbuild.Preconditions{
			TimeBounds: txnbuild.NewInfiniteTimeout(),
		},
	}

	tx, err := txnbuild.NewTransaction(txParams)
	if err != nil {
		return "", 0, fmt.Errorf("error building transaction: %v", err)
	}

	signedTx, err := tx.Sign(w.networkPassphrase, kp)
	if err != nil {
		return "", 0, fmt.Errorf("error signing transaction: %v", err)
	}

	resp, err := w.client.SubmitTransaction(signedTx)
	if err != nil {
		return "", 0, fmt.Errorf("error submitting transaction: %v", err)
	}

	if !resp.Successful {
		return "", 0, fmt.Errorf("transaction failed: %s", resp.ResultXdr)
	}

	return resp.Hash, amountFloat, nil
}