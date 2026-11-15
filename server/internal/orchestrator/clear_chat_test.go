package orchestrator

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/ws"
)

func TestClearSessionChatResetsInMemoryHistoryAndTruncatesPersistedMessages(t *testing.T) {
	store, err := character.NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, err := store.Create(&character.Character{
		Name:          "Clear",
		AvatarBackend: character.AvatarBackendLocalImage,
		SystemPrompt:  "You are helpful.",
	})
	if err != nil {
		t.Fatal(err)
	}

	mgr := NewSessionManager(4)
	session, err := mgr.Create("clear-session", ModeStandard, char.ID)
	if err != nil {
		t.Fatal(err)
	}

	startedAt := time.Now().UTC()
	if err := store.SaveConversation(char.ID, session.ID, startedAt, startedAt, []map[string]any{
		{"role": "user", "content": "remember me", "timestamp": startedAt.Format(time.RFC3339Nano)},
	}); err != nil {
		t.Fatal(err)
	}
	session.SetDialogContext([]DialogContextItem{{Role: "user", Text: "remember me", Timestamp: 1}})
	session.AddMessage(ChatMessage{Role: "user", Content: "remember me"})
	session.AddMessage(ChatMessage{Role: "assistant", Content: "I remember"})

	orch := New(nil, ws.NewHub(), mgr, nil, store)
	clearedAt, err := orch.ClearSessionChat(session.ID)
	if err != nil {
		t.Fatal(err)
	}
	if clearedAt.IsZero() {
		t.Fatal("expected cleared_at")
	}
	if len(session.HistorySnapshot()) != 0 {
		t.Fatalf("expected empty history after clear, got %v", session.HistorySnapshot())
	}
	if len(session.DialogContextSnapshot()) != 0 {
		t.Fatal("expected dialog context to be cleared")
	}

	loaded, _, _, err := store.LoadRecentMessages(char.ID, "", 50)
	if err != nil {
		t.Fatal(err)
	}
	for _, msg := range loaded {
		if content, _ := msg["content"].(string); content == "remember me" {
			t.Fatalf("persisted session messages should be truncated, got %+v", loaded)
		}
	}

	gotChar, err := store.Get(char.ID)
	if err != nil {
		t.Fatal(err)
	}
	if gotChar.ChatHistoryClearedAt == "" {
		t.Fatal("expected character chat_history_cleared_at after clear")
	}

	recDir := store.SessionRecordingDir(char.ID, session.ID, session.CreatedAt)
	turnPath := filepath.Join(recDir, "turn1.mp4")
	if err := os.WriteFile(turnPath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := orch.ClearSessionChat(session.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(turnPath); err != nil {
		t.Fatalf("turn recording should remain after clear: %v", err)
	}
}