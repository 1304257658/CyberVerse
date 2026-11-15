package character

import (
	"testing"
	"time"
)

func TestSetChatHistoryClearedAtFiltersLoadRecentMessagesAcrossSessions(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, err := store.Create(&Character{Name: "Filter", AvatarBackend: AvatarBackendLocalImage})
	if err != nil {
		t.Fatal(err)
	}

	oldSession := "11111111-1111-1111-1111-111111111111"
	newSession := "22222222-2222-2222-2222-222222222222"
	t0 := time.Date(2026, 11, 15, 10, 0, 0, 0, time.UTC)
	t1 := t0.Add(time.Minute)

	if err := store.SaveConversation(char.ID, oldSession, t0, t0, []map[string]any{
		{"role": "user", "content": "before clear", "timestamp": t0.Format(time.RFC3339Nano)},
	}); err != nil {
		t.Fatal(err)
	}
	clearedAt := t1
	if err := store.SetChatHistoryClearedAt(char.ID, "", clearedAt); err != nil {
		t.Fatal(err)
	}
	if err := store.SaveConversation(char.ID, newSession, clearedAt.Add(time.Second), clearedAt.Add(2*time.Second), []map[string]any{
		{"role": "user", "content": "after clear", "timestamp": clearedAt.Add(time.Second).Format(time.RFC3339Nano)},
	}); err != nil {
		t.Fatal(err)
	}

	loaded, _, _, err := store.LoadRecentMessages(char.ID, "", 50)
	if err != nil {
		t.Fatal(err)
	}
	var contents []string
	for _, msg := range loaded {
		if c, _ := msg["content"].(string); c != "" {
			contents = append(contents, c)
		}
	}
	if len(contents) != 1 || contents[0] != "after clear" {
		t.Fatalf("expected only post-clear message, got %+v", contents)
	}

	reloaded, err := store.Get(char.ID)
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.ChatHistoryClearedAt == "" {
		t.Fatal("expected chat_history_cleared_at on character")
	}
}