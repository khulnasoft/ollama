package parser

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
)

type Command struct {
	Name string
	Args string
	bytes.Buffer
}

const (
	stateName = iota
	stateArgs
	stateMultiline
	stateParameter
	stateMessage
	stateComment
)

func Parse(r io.Reader) ([]Command, error) {
	var cmds []Command
	var cmd Command
	var b bytes.Buffer

	s := stateName
	br := bufio.NewReader(r)
	for {
		r, _, err := br.ReadRune()
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}

		if _, err := cmd.WriteRune(r); err != nil {
			return nil, err
		}

		switch s {
		case stateName:
			switch {
			case alpha(r), number(r):
				if _, err := b.WriteRune(r); err != nil {
					return nil, err
				}
			case r == '#':
				s = stateComment
			case space(r):
				if b.Len() > 0 {
					cmd.Name = strings.ToLower(b.String())
					b.Reset()

					if cmd.Name == "from" {
						cmd.Name = "model"
					}

					switch cmd.Name {
					case "parameter":
						s = stateParameter
					case "message":
						s = stateMessage
					default:
						s = stateArgs
					}
				}
			case newline(r):
				if b.Len() > 0 {
					return nil, fmt.Errorf("missing value for [%s]", b.String())
				}
			default:
				return nil, fmt.Errorf("unexpected rune %q for state %d", r, s)
			}
		case stateParameter:
			switch {
			case alpha(r), number(r), r == '_':
				if _, err := b.WriteRune(r); err != nil {
					return nil, err
				}
			case space(r):
				cmd.Name = strings.ToLower(b.String())
				b.Reset()
				s = stateArgs
			case newline(r):
				return nil, fmt.Errorf("missing value for [%s]", b.String())
			default:
				return nil, fmt.Errorf("unexpected rune %q for state %d", r, s)
			}
		case stateArgs:
			switch {
			// TODO
			// case quote(r):
			case newline(r):
				cmd.Args += b.String()
				b.Reset()

				cmds = append(cmds, cmd)
				cmd = Command{}
				s = stateName
			default:
				if _, err := b.WriteRune(r); err != nil {
					return nil, err
				}
			}
		case stateMultiline:
			switch {
			// TODO
			// case quote(r):
			default:
				if _, err := b.WriteRune(r); err != nil {
					return nil, err
				}
			}
		case stateMessage:
			switch {
			case space(r):
				if !isValidRole(b.String()) {
					return nil, errors.New("role must be one of \"system\", \"user\", or \"assistant\"")
				}

				if _, err := b.WriteString(": "); err != nil {
					return nil, err
				}

				cmd.Args += b.String()
				b.Reset()
				s = stateArgs
			case newline(r):
				return nil, fmt.Errorf("missing value for [%s]", b.String())
			default:
				if _, err := b.WriteRune(r); err != nil {
					return nil, err
				}
			}
		case stateComment:
			if newline(r) {
				b.Reset()
				cmd = Command{}
				s = stateName
			}
		}
	}

	// handle trailing buffer
	if b.Len() > 0 {
		switch s {
		case stateArgs:
			cmd.Args = b.String()
			cmds = append(cmds, cmd)
		case stateMultiline:
			return nil, errors.New("unterminated multiline string")
		default:
			return nil, fmt.Errorf("missing value for [%s]", b.String())
		}
	}

	for _, cmd := range cmds {
		if cmd.Name == "model" {
			return cmds, nil
		}
	}

	return nil, errors.New("no FROM line")
}

func alpha(r rune) bool {
	return r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z'
}

func number(r rune) bool {
	return r >= '0' && r <= '9'
}

func space(r rune) bool {
	return r == ' ' || r == '\t'
}

func newline(r rune) bool {
	return r == '\r' || r == '\n'
}

func quote(r rune) bool {
	return r == '"' || r == '\''
}

func isValidRole(role string) bool {
	return role == "system" || role == "user" || role == "assistant"
}
