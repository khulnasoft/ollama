package parser

import (
	"bytes"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
)

var ignoreCommandBuffer = cmpopts.IgnoreFields(Command{}, "Buffer")

func TestParser(t *testing.T) {

	input := `
FROM model1
ADAPTER adapter1
LICENSE MIT
PARAMETER param1 value1
PARAMETER param2 value2
TEMPLATE template1
`

	reader := strings.NewReader(input)

	commands, err := Parse(reader)
	assert.Nil(t, err)

	expectedCommands := []Command{
		{Name: "model", Args: "model1"},
		{Name: "adapter", Args: "adapter1"},
		{Name: "license", Args: "MIT"},
		{Name: "param1", Args: "value1"},
		{Name: "param2", Args: "value2"},
		{Name: "template", Args: "template1"},
	}

	assert.True(t, cmp.Equal(expectedCommands, commands, ignoreCommandBuffer))
}

func TestParserNoFromLine(t *testing.T) {

	input := `
PARAMETER param1 value1
PARAMETER param2 value2
`

	reader := strings.NewReader(input)

	_, err := Parse(reader)
	assert.ErrorContains(t, err, "no FROM line")
}

func TestParserMissingValue(t *testing.T) {

	input := `
FROM foo
PARAMETER param1
`

	reader := strings.NewReader(input)

	_, err := Parse(reader)
	assert.ErrorContains(t, err, "missing value for [param1]")

}

func TestParserMessages(t *testing.T) {

	input := `
FROM foo
MESSAGE system You are a Parser. Always Parse things.
MESSAGE user Hey there!
MESSAGE assistant Hello, I want to parse all the things!
`

	reader := strings.NewReader(input)
	commands, err := Parse(reader)
	assert.Nil(t, err)

	expectedCommands := []Command{
		{Name: "model", Args: "foo"},
		{Name: "message", Args: "system: You are a Parser. Always Parse things."},
		{Name: "message", Args: "user: Hey there!"},
		{Name: "message", Args: "assistant: Hello, I want to parse all the things!"},
	}

	assert.True(t, cmp.Equal(expectedCommands, commands, ignoreCommandBuffer))
}

func TestParserMessagesBadRole(t *testing.T) {

	input := `
FROM foo
MESSAGE badguy I'm a bad guy!
`

	reader := strings.NewReader(input)
	_, err := Parse(reader)
	assert.ErrorContains(t, err, "role must be one of \"system\", \"user\", or \"assistant\"")
}

func TestParserMultiline(t *testing.T) {
	var cases = []struct {
		input    string
		expected []Command
	}{
		{
			`FROM foo
TEMPLATE """
{{ .System }}

{{ .Prompt }}
"""

SYSTEM """
This is a multiline system message.
"""
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "{{ .System }}\n\n{{ .Prompt }}\n"},
				{Name: "system", Args: "This is a multiline system message.\n"},
			},
		},
		{
			`FROM foo
TEMPLATE """{{ .System }} {{ .Prompt }}"""`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "{{ .System }} {{ .Prompt }}"},
			},
		},
	}

	for _, tc := range cases {
		t.Run("", func(t *testing.T) {
			reader := strings.NewReader(tc.input)
			commands, err := Parse(reader)
			assert.Nil(t, err)

			assert.True(t, cmp.Equal(tc.expected, commands, ignoreCommandBuffer))
		})
	}
}

func TestParserParameters(t *testing.T) {

	var cases = []string{
		"numa true",
		"num_ctx 1",
		"num_batch 1",
		"num_gqa 1",
		"num_gpu 1",
		"main_gpu 1",
		"low_vram true",
		"f16_kv true",
		"logits_all true",
		"vocab_only true",
		"use_mmap true",
		"use_mlock true",
		"num_thread 1",
		"num_keep 1",
		"seed 1",
		"num_predict 1",
		"top_k 1",
		"top_p 1.0",
		"tfs_z 1.0",
		"typical_p 1.0",
		"repeat_last_n 1",
		"temperature 1.0",
		"repeat_penalty 1.0",
		"presence_penalty 1.0",
		"frequency_penalty 1.0",
		"mirostat 1",
		"mirostat_tau 1.0",
		"mirostat_eta 1.0",
		"penalize_newline true",
		"stop foo",
	}

	for _, c := range cases {
		t.Run(c, func(t *testing.T) {
			var b bytes.Buffer
			b.WriteString("FROM foo\nPARAMETER ")
			b.WriteString(c)
			_, err := Parse(&b)
			assert.Nil(t, err)
		})
	}
}

func TestParserOnlyFrom(t *testing.T) {
	commands, err := Parse(strings.NewReader("FROM foo"))
	assert.Nil(t, err)

	expected := []Command{{Name: "model", Args: "foo"}}
	assert.True(t, cmp.Equal(expected, commands, ignoreCommandBuffer))
}
