package parser

import (
	"bytes"
	"errors"
	"fmt"
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

func TestParserParametersMissingValue(t *testing.T) {

	input := `
FROM foo
PARAMETER param1
`

	reader := strings.NewReader(input)

	_, err := Parse(reader)
	assert.ErrorContains(t, err, "missing value for [param1]")
}

func TestParserMessages(t *testing.T) {
	var cases = []struct {
		input    string
		expected []Command
		err      error
	}{
		{
			`
FROM foo
MESSAGE system You are a Parser. Always Parse things.
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "message", Args: "system: You are a Parser. Always Parse things."},
			},
			nil,
		},
		{
			`
FROM foo
MESSAGE system You are a Parser. Always Parse things.
MESSAGE user Hey there!
MESSAGE assistant Hello, I want to parse all the things!
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "message", Args: "system: You are a Parser. Always Parse things."},
				{Name: "message", Args: "user: Hey there!"},
				{Name: "message", Args: "assistant: Hello, I want to parse all the things!"},
			},
			nil,
		},
		{
			`
FROM foo
MESSAGE system """
You are a multiline Parser. Always Parse things.
"""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "message", Args: "system: \nYou are a multiline Parser. Always Parse things.\n"},
			},
			nil,
		},
		{
			`
FROM foo
MESSAGE badguy I'm a bad guy!
`,
			nil,
			errors.New("role must be one of \"system\", \"user\", or \"assistant\""),
		},
		{
			`
FROM foo
MESSAGE system
`,
			nil,
			errors.New("missing value for [system]"),
		},
		{
			`
FROM foo
MESSAGE system`,
			nil,
			errors.New("missing value for [system]"),
		},
	}

	for _, c := range cases {
		t.Run("", func(t *testing.T) {
			commands, err := Parse(strings.NewReader(c.input))
			assert.Equal(t, err, c.err)
			assert.Truef(t, cmp.Equal(c.expected, commands, ignoreCommandBuffer), "expected %#v, got %#v", c.expected, commands)
		})
	}
}

func TestParserMultiline(t *testing.T) {
	var cases = []struct {
		multiline string
		expected  []Command
		err       error
	}{
		{
			`
FROM foo
TEMPLATE """
This is a
multiline template.
"""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "\nThis is a\nmultiline template.\n"},
			},
			nil,
		},
		{
			`
FROM foo
TEMPLATE """
This is a
multiline template."""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "\nThis is a\nmultiline template."},
			},
			nil,
		},
		{
			`
FROM foo
TEMPLATE """This is a
multiline template."""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "This is a\nmultiline template."},
			},
			nil,
		},
		{
			`
FROM foo
TEMPLATE """This is a multiline template."""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "This is a multiline template."},
			},
			nil,
		},
		{
			`
FROM foo
TEMPLATE """This is a multiline template.""
			`,
			nil,
			errors.New("unterminated multiline string"),
		},
		{
			`
FROM foo
TEMPLATE """
This is a multiline template with "quotes".
"""
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "\nThis is a multiline template with \"quotes\".\n"},
			},
			nil,
		},
	}

	for _, c := range cases {
		t.Run("", func(t *testing.T) {
			commands, err := Parse(strings.NewReader(c.multiline))
			assert.Equal(t, err, c.err)
			assert.Truef(t, cmp.Equal(c.expected, commands, ignoreCommandBuffer), "expected %#v, got %#v", c.expected, commands)
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
			fmt.Fprintln(&b, "FROM foo")
			fmt.Fprintln(&b, "PARAMETER", c)
			t.Logf("input: %s", b.String())
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

func TestParserComments(t *testing.T) {
	var cases = []struct {
		input    string
		expected []Command
	}{
		{
			`
# comment
FROM foo
	`,
			[]Command{
				{Name: "model", Args: "foo"},
			},
		},
	}

	for _, c := range cases {
		t.Run("", func(t *testing.T) {
			commands, err := Parse(strings.NewReader(c.input))
			assert.Nil(t, err)
			assert.True(t, cmp.Equal(c.expected, commands, ignoreCommandBuffer))
		})
	}
}
