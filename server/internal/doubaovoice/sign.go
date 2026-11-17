package doubaovoice

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const timeFormatV4 = "20060102T150405Z"

type credentials struct {
	accessKeyID     string
	secretAccessKey string
	service         string
	region          string
}

func signVolcRequest(req *http.Request, body []byte, cred credentials) error {
	if cred.accessKeyID == "" || cred.secretAccessKey == "" {
		return fmt.Errorf("volc credentials missing")
	}
	formatDate := time.Now().UTC().Format(timeFormatV4)
	date := formatDate[:8]
	host := req.URL.Host
	if host == "" {
		host = req.Host
	}
	contentType := "application/json; charset=UTF-8"
	bodyHash := hashSHA256(body)

	canonicalHeaders := fmt.Sprintf("content-type:%s\nhost:%s\nx-content-sha256:%s\nx-date:%s\n", contentType, host, bodyHash, formatDate)
	signedHeaders := "content-type;host;x-content-sha256;x-date"
	canonicalRequest := strings.Join([]string{
		req.Method,
		normalizeURI(req.URL.Path),
		normalizedQuery(req.URL.Query()),
		canonicalHeaders,
		signedHeaders,
		bodyHash,
	}, "\n")
	hashedCanon := hashSHA256([]byte(canonicalRequest))
	credentialScope := strings.Join([]string{date, cred.region, cred.service, "request"}, "/")
	stringToSign := strings.Join([]string{
		"HMAC-SHA256",
		formatDate,
		credentialScope,
		hashedCanon,
	}, "\n")
	signingKey := signingKeyV4(cred.secretAccessKey, date, cred.region, cred.service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, stringToSign))
	auth := fmt.Sprintf(
		"HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		cred.accessKeyID,
		credentialScope,
		signedHeaders,
		signature,
	)

	req.Header.Set("Host", host)
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("X-Date", formatDate)
	req.Header.Set("X-Content-Sha256", bodyHash)
	req.Header.Set("Authorization", auth)
	return nil
}

func signingKeyV4(secretKey, date, region, service string) []byte {
	kDate := hmacSHA256([]byte(secretKey), date)
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, service)
	return hmacSHA256(kService, "request")
}

func hmacSHA256(key []byte, content string) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(content))
	return mac.Sum(nil)
}

func hashSHA256(content []byte) string {
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:])
}

func normalizedQuery(v url.Values) string {
	return strings.ReplaceAll(v.Encode(), "+", "%20")
}

func normalizeURI(uri string) string {
	if uri == "" {
		return "/"
	}
	if !strings.HasPrefix(uri, "/") {
		uri = "/" + uri
	}
	return uri
}

func newSignedPost(ctx credentials, action, version string, payload []byte) (*http.Request, error) {
	rawURL := fmt.Sprintf("https://open.volcengineapi.com/?Action=%s&Version=%s", url.QueryEscape(action), url.QueryEscape(version))
	req, err := http.NewRequest(http.MethodPost, rawURL, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	if err := signVolcRequest(req, payload, ctx); err != nil {
		return nil, err
	}
	return req, nil
}

func readBody(req *http.Request) ([]byte, error) {
	if req.Body == nil {
		return nil, nil
	}
	defer req.Body.Close()
	return io.ReadAll(req.Body)
}
