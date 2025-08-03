package wallet

import (
	"fmt"
	"strconv"

	"github.com/stellar/go/keypair"
	"github.com/stellar/go/txnbuild"
)

// Enhanced wallet methods with dynamic fee support

func (w *Wallet) TransferWithDynamicFee(kp *keypair.Full, amountStr string, address string, dynamicFee float64) error {
	account, err := w.GetAccount(kp)
	if err != nil {
		return fmt.Errorf("error getting account: %w", err)
	}

	amount, err := strconv.ParseFloat(amountStr, 64)
	if err != nil {
		return fmt.Errorf("invalid amount: %w", err)
	}

	paymentOp := txnbuild.Payment{
		Destination: address,
		Amount:      strconv.FormatFloat(amount, 'f', -1, 64),
		Asset:       txnbuild.NativeAsset{},
	}

	// Use dynamic fee instead of fixed fee
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

	// Use dynamic fee for competitive advantage
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