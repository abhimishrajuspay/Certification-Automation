# Evidence-derived recipes — live local newton-hs + npci-mocking certification stack

Source: the 2026-08-31→09-01 live run validating BCRP-Comfort-v1 against
`/Newton-hs` @ `Feature/uat-simulation-toggle-api` (v3.5.95) + the user's
`npci-mocking` @ 8089 + postgres `amazon-upi` @ 5432 + Redis @ 6379.

## The DB-error → remedy ladder (in order of discovery)

### 1. `LOCKED_OUT (checksum bypass)` — before anything else
Any merchant S2S route fails until the merchant record can pass
`MerchantConfigurations.checksumBypassId` comparison:

```sql
UPDATE "MerchantConfigurations"
SET active = true, "checksumBypassId" = 'localcert'
WHERE "MerchantId" = '<merchantId>';
```

Watch out: the signald on every new deploy — DB seeds may have `active IS NULL`;
the earlier `ALTER TABLE "MerchantConfigurations" ADD COLUMN "active" BOOLEAN NOT NULL
DEFAULT TRUE` comes from `db_schema/configdb.sql:49` on the branch.

### 2. `INTERNAL / "/upi/..." 404` — NPCI_DOMAIN double-slash
`.env.local`:

```
NPCI_DOMAIN=http://localhost:8089        # NO trailing slash
```

If `NPCI_DOMAIN` ends with `/`, the assembled URL is `//upi/ReqValAdd/…` → 404.

### 3. `SERVICE_UNAVAILABLE` — mocker not running
NPCI makes a real outbound HTTP call. Root process: `npm run dev` in the
`npci-mocking/` clone; correctness marker: `curl -s -m 3 -o /dev/null http://localhost:8089/`
→ a 404 response body (server up), NOT `000` (down).

### 4. `INVALID_DATA / User profile not found` — merchant customer step
For S2S customer-bound flows the middleware resolves
`MerchantCustomers(merchantId, merchantCustomerId)`. Narrow insert:

```sql
INSERT INTO "Customers" (id, "createdAt", "updatedAt", active, "mobileNumber")
VALUES ('LC-CUST-0001', now(), now(), true, '9876543210');

INSERT INTO "MerchantCustomers"
  (id, "merchantCustomerId", "mobileNumber", "createdAt", "updatedAt",
   active, "MerchantId", "CustomerId", secretKey, store, "packageName",
   email, "mobileNumberHash")
VALUES ('LC-0001', 'LOCAL-CERT-CUST-01', '9876543210', now(), now(), true,
        '<merchant-db-id>', 'LC-CUST-0001', 'LOCALSECRET', '{}'::json,
        'in.juspay.merchant', 'local@cert.dev',
        '<sha256 of mobile number>');
```

### 5. `No active device binding for merchantCustomer`
`MerchantCustomers.DeviceId` must exist; `Devices` row needed (fingerprint recipe in §8).

### 6. `value is missing for the key in function: … Name not found`
Customer profile resolved but name unavailable. Insert, IF ABSENT, a
config row that Davies the `getNameFromAccount` path, THEN clear redis configs:

```sql
INSERT INTO "Configurations" ("key","value","createdAt","updatedAt")
VALUES ('fallbackToMobileNumberForVpaResolution','true',now(),now())
ON CONFLICT ("key") DO UPDATE SET "value"='true';
```
`redis-cli SCAN… newton-configurationJson | xargs -n1 redis-cli DEL`

### 7. `Account not found(vpaccountHash)` — no linkage
```sql
INSERT INTO "Accounts"
  (id, "accountNumber", "maskedAccountNumber", ifsc, "mpinSet",
   "aadharEnabled", type, name, "bankCode", "bankName", "CustomerId",
   "accountHash", "credsAllowed", "createdAt", "updatedAt", active, "default",
   "accSubType")
VALUES ('LC-ACC-0001', '1234567890xx', 'xxxxxx', 'AABC0019', false, false,
        'SAVINGS', 'LOCAL CERT CUSTOMER', 'AABC', 'NPCI MOCK BANK',
        'LC-CUST-0001', 'BAULOC0001', '[COMP]', now(), now(), true, true,
        'SAVING');
INSERT INTO "MerchantCustomerAccounts"
  (id, "AccountId", "MerchantCustomerId", "default", "createdAt",
   "updatedAt", active, "accountHash")
VALUES ('LC-MCA-0001', 'LC-ACC-0001', 'LC-0001', true, now(), now(), true,
        'BAULOC0001');
```
Note: `bankAccountUniqueId` => PLAIN `Accounts.accountHash` equal (NOT hashed).

### 8. DEVICE_FINGERPRINT_MISMATCH
The wire request `deviceFingerPrint` must equal `sha256(fingerprint <> ssid)`
of a `Devices` row whose `id` is `MerchantCustomers.DeviceId`:

```sql
INSERT INTO "Devices" (id, os, fingerprint, ssid, model, version,
                        manufacturer, "createdAt", "updatedAt",
                        "fingerprintHash", "ssidHash")
VALUES ('LC-DEV-0001', 'android', 'lcrawfp0001', '1234', 'UPICertDevice',
        '13', 'Juspay', now(), now(), md5('lcrawfp0001'), md5('1234'));
```
For fingerprint `lcrawfp0001` + ssid `1234`, wire value is
`686afe4e2ee4a5a25319e9c3a7e1ef43c583629c0ad0d144b89f40b37613b691`.
Compute dynamically via `Crypto.sha256Hash $ fingerprint' <> ssid'`
(`src/Newton/Utils/Transformers/Transformer.hs:1057`, mkDeviceFingerprint).

### 9. `INVALID_DATA: vpaHandlerSync refresh` — `NPCI_HANDLE` conflict
`.env.local`:

```
NPCI_HANDLE=@vpa        # must match the SHEET's vpa suffix
```
without this, `validateVpaFormat` rejects all `mtNN@vpa` VPAs. The merchant
store's `vpaDomain` is a lower-priority override set via:
```sql
UPDATE "Merchants" SET store=(store::jsonb || '{"vpaDomain":"@vpa"}'::jsonb)::json
WHERE id='<merchantDbId>';
```
requires redis `newton-*TESTMERCHANT*` key invalidation to take effect.

### 10. `column t0.<x> does not exist` — stale dump drift
Pattern: local dump predates recent branch migrations. Fix precisely per
branch source of truth, grep `db_schema/changelog.sql` for the ADD VALUE:
e.g. `ALTER TYPE "enum_Vpas_status" ADD VALUE IF NOT EXISTS 'DEBIT_BLOCKED';` /
`'CREDIT_BLOCKED'` / `'FASTAG_RESTRICTED_PAY'` (lines 522,523,1150).

### 11. `Row not found: MerchantCustomerAccounts table`
Either reseed the LC-MCA-0001 row (§7), or the branch advertised the new
`linkedVpasCount` column earlier than yours: `ALTER TABLE "MerchantCustomerAccounts"
ADD COLUMN IF NOT EXISTS "linkedVpasCount" integer;`.

### 12. NPCI-mock per-VPA profiles (know the difference from newton issues)
`npci-mocking` ships with fixed in-code per-VPA fixtures. During folder-02 we
observed `mt13@vpa` = `gatewayResponseCode "00"` (balance 31.99) and every
`mt14..mt21` = `"U01"`. That is NOT a newton defect; it's the mock's
pre-installed profile set. The `cbs` config-side fixtures live in code (e.g.
 `npci-mocking/src/data/*`/`dummyBankAccounts`). When the sheet expects a
scenario-specific code (XH/PM0/ZG/MM2…) the operator must first install the
mock fixture; the skill stops flagging API-side defects if EXPECTED-vs-ACTUAL
differs ONLY at `gatewayResponseCode` values set by case data. ALWAYS record
both envelope status AND `gatewayResponseCode` on the folder evidence ledger.

## Wire shapes (from observed successful exchanges)

- `x-timestamp`: 13-digit ms epoch string (NOT ISO) for the merchant middleware.
- Body `iat`: same value in string form ("1769900000000").
- `UPICredBlock` (embedded string inside `credBlock` field):
  ```json
  {"mpincred":{"type":"PIN","subType":"MPIN","data":{"type":"","skey":"","pid":"","ki":"20150822","hmac":"","encryptedBase64String":"2.0|DUMMYBASE64STRING0000000000000000000000==","code":"NPCI"}}}
  ```
  For SET/RESET-PIN list `otpcred`, `atmpincred`, `newcred` + `card: 6-digit` +
  `expiry: 4-char` alongside.
- Balance requests need `bankAccountUniqueId` (plain Account.accountHash) + vpas
  seeds row in `Vpas`, with `MerchantCustomerId = <mcId>`.

## Red flags (stop & read before more retries)
1. The same error for three consecutive requests — the remediation ladder is
   wrong; inspect `newton-local.log` for the `category=="DB"` row first.
2. Redis shows a key for the previous run — delete before re-testing the SAME
   family.
3. A new folder switching families before ChkTxn's pushes — the `merchantRequestId`
   needs to be real (from a previous Pay response `gatewayReferenceId`).

## Ownership contract
- LC- prefix: everything I inserted. Safe to drop after the stack round.
- No `Vacuum`/`Reindex`/`Mass delete` commands during live certification runs.
- NEVER commit `.env.local` — keys are per-machine; use `git stash pop` to get
  the operator's pending branch WIP back before handing over the machine.
