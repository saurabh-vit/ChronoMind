"""
Conversation parser for ChronoMind.

Reads a file containing conversations as double-quote-delimited
text blocks. Each block is one independent conversation between
User 1 and User 2. Parses individual messages and returns a flat,
chronologically ordered list of Message objects with conversation IDs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .models import Message

logger = logging.getLogger(__name__)

# Pattern to match "User N: message text"
_MESSAGE_PATTERN = re.compile(
    r"^(User\s*\d+)\s*:\s*(.+)$", re.IGNORECASE
)


def parse_conversations(csv_path: str | Path) -> List[Message]:
    """
    Parse a conversations file into a chronological list of Messages.

    The file contains conversations as double-quote-delimited blocks.
    Each block is an independent conversation. Messages within each
    block are separated by newlines in "User N: text" format.

    There is NO CSV header row.

    Args:
        csv_path: Path to the conversations file.

    Returns:
        Flat list of Message objects in chronological order,
        each tagged with a conversation_id.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Conversations file not found: {csv_path}")

    logger.info("Parsing conversations from %s", csv_path)

    # Read the entire file
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        raw_data = f.read()

    # Extract conversation blocks: everything between double quotes
    conversation_blocks = _extract_conversation_blocks(raw_data)

    logger.info("Found %d conversation blocks.", len(conversation_blocks))

    if not conversation_blocks:
        raise ValueError("No conversation blocks found in file.")

    # Parse each conversation block into messages
    messages: List[Message] = []
    global_index = 0
    skipped = 0

    for conv_id, block in enumerate(conversation_blocks):
        block = block.strip()
        if not block:
            skipped += 1
            continue

        # Handle escaped quotes: "" -> "
        block = block.replace('""', '"')

        # Split into individual lines (messages)
        lines = block.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = _MESSAGE_PATTERN.match(line)
            if match:
                user = match.group(1).strip()
                text = match.group(2).strip()
            else:
                # Non-matching line: append to previous message as continuation
                if messages and messages[-1].conversation_id == conv_id:
                    messages[-1].text += " " + line
                    messages[-1].raw_line += " " + line
                    continue
                else:
                    # Orphan line at start of conversation — skip
                    continue

            if text:
                msg = Message(
                    index=global_index,
                    user=user,
                    text=text,
                    conversation_id=conv_id,
                    raw_line=line,
                )
                messages.append(msg)
                global_index += 1

    logger.info(
        "Parsed %d messages from %d conversations (%d empty blocks skipped).",
        len(messages),
        len(conversation_blocks) - skipped,
        skipped,
    )
    return messages


def _extract_conversation_blocks(raw_data: str) -> List[str]:
    """
    Extract conversation blocks from the raw file data.

    Each conversation is wrapped in double quotes. Handles:
    - Newlines within quoted blocks
    - Escaped quotes ("") inside text
    - Various line endings (LF, CRLF)

    Uses a state-machine approach for robust parsing.
    """
    blocks: List[str] = []
    current_block: List[str] = []
    in_block = False
    i = 0
    data_len = len(raw_data)

    while i < data_len:
        char = raw_data[i]

        if not in_block:
            if char == '"':
                in_block = True
                current_block = []
                i += 1
                continue
            # Skip characters between blocks
            i += 1
            continue

        # Inside a quoted block
        if char == '"':
            # Check for escaped quote ""
            if i + 1 < data_len and raw_data[i + 1] == '"':
                current_block.append('""')
                i += 2
                continue
            else:
                # End of block
                blocks.append("".join(current_block))
                in_block = False
                i += 1
                continue

        current_block.append(char)
        i += 1

    # Handle unclosed block at end of file
    if in_block and current_block:
        blocks.append("".join(current_block))

    return blocks


def get_conversation_groups(messages: List[Message]) -> Dict[int, List[Message]]:
    """
    Group messages by conversation_id.

    Args:
        messages: Flat list of all messages.

    Returns:
        Dict mapping conversation_id to list of messages in that conversation.
    """
    groups: Dict[int, List[Message]] = {}
    for msg in messages:
        if msg.conversation_id not in groups:
            groups[msg.conversation_id] = []
        groups[msg.conversation_id].append(msg)
    return groups


def get_user_names(messages: List[Message]) -> List[str]:
    """Extract unique user names from a message list, preserving order."""
    seen = set()
    names = []
    for msg in messages:
        if msg.user not in seen:
            seen.add(msg.user)
            names.append(msg.user)
    return names


def get_conversation_count(messages: List[Message]) -> int:
    """Get the number of unique conversations."""
    return len(set(m.conversation_id for m in messages))


def format_messages(messages: List[Message]) -> str:
    """Format a list of messages into a readable text block."""
    return "\n".join(str(m) for m in messages)
