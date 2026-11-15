package character

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestClearPersistedSessionMessagesEmptiesSessionJSONKeepsRecordings(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, err := store.Create(&Character{Name: "Test", AvatarBackend: AvatarBackendLocalImage})
	if err != nil {
		t.Fatal(err)
	}

	sessionID := "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
	startedAt := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	messages := []map[string]any{
		{"role": "user", "content": "hello", "timestamp": startedAt.Format(time.RFC3339Nano)},
		{"role": "assistant", "content": "hi", "timestamp": startedAt.Add(time.Second).Format(time.RFC3339Nano)},
	}
	if err := store.SaveConversation(char.ID, sessionID, startedAt, startedAt.Add(time.Minute), messages); err != nil {
		t.Fatal(err)
	}
	recDir := store.SessionRecordingDir(char.ID, sessionID, startedAt)
	turnFile := filepath.Join(recDir, "turn1.txt")
	if err := os.WriteFile(turnFile, []byte("transcript"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := store.ClearPersistedSessionMessages(char.ID, "", sessionID); err != nil {
		t.Fatal(err)
	}

	sessionJSON := filepath.Join(recDir, "session.json")
	data, err := os.ReadFile(sessionJSON)
	if err != nil {
		t.Fatal(err)
	}
	var record struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatal(err)
	}
	if len(record.Messages) != 0 {
		t.Fatalf("expected empty messages after clear, got %d", len(record.Messages))
	}
	if _, err := os.Stat(turnFile); err != nil {
		t.Fatalf("expected turn recording to remain: %v", err)
	}

	loaded, _, _, err := store.LoadRecentMessages(char.ID, "", 50)
	if err != nil {
		t.Fatal(err)
	}
	for _, msg := range loaded {
		if msg["content"] == "hello" || msg["content"] == "hi" {
			t.Fatalf("cleared messages should not load from disk: %+v", loaded)
		}
	}
}