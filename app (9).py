"""
Enhanced Discord Bot with Mini-Games, AI Features, and Holiday Messaging

Features:
- Custom Commands creation with AI (Gemini 2.5 Flash)
- Local file storage in serverdata folder
- Help command with AI assistance
- Mini-Games: Hangman, Tic-Tac-Toe, RPG Adventure, Eject Intruder, Word Scramble
- Fun features: Cat pictures, Fun Facts
- AI Chat features: Codex Agent, Basic Chat (all ephemeral)
- Holiday messaging with PIL postcards
- Halloween haunted messages (Oct 20-30)
- Dev notification system
"""

import os
import sys
import logging
import asyncio
import inspect
import ast
import textwrap
import traceback
import re
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from io import BytesIO

# Add the project root to Python path to find local modules
# Use multiple methods to ensure it works across different environments
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Also add current working directory to be safe
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv
load_dotenv()

import discord
from discord.ext import commands, tasks
from discord import app_commands
from discord.ui import View, Button, Select, Modal, TextInput
import aiohttp
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import requests

from google import genai
from google.genai import types
import openai

from data import (
    HANGMAN_WORDS, WORD_SCRAMBLE_WORDS, FUN_FACTS_LIST,
    RPG_SCENARIOS_LIST, INTRUDER_PATTERNS_LIST, EMOJI_CATEGORIES,
    HAUNTED_MESSAGES
)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN") or os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEV_USER_ID = 1288144571494170706
LOG_SERVER_ID = 1429484464781922326
LOG_CHANNEL_ID = 1443933806431047795
STORAGE_SERVER_ID = 1437347245933596706
STORAGE_DATA_CHANNEL_ID = 1437347246541635606
STORAGE_VARIABLES_CHANNEL_ID = 1445738633528086619

if not DISCORD_TOKEN:
    print("ERROR: Missing DISCORD_TOKEN")
    sys.exit(1)

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


CONFIG_FILE = "config/data.json"
DATA_FILE = "serverdata/data.json"
data_lock = asyncio.Lock()
remote_storage_message_id = None
remote_variables_message_id = None


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

async def get_remote_storage_channel():
    """Get the remote storage channel"""
    try:
        guild = bot.get_guild(STORAGE_SERVER_ID)
        if not guild:
            log.error(f"Storage server {STORAGE_SERVER_ID} not found")
            return None
        channel = guild.get_channel(STORAGE_DATA_CHANNEL_ID)
        if not channel:
            log.error(f"Storage channel {STORAGE_DATA_CHANNEL_ID} not found")
        return channel
    except Exception as e:
        log.error(f"Error getting storage channel: {e}")
        return None

async def get_remote_variables_channel():
    """Get the remote variables channel"""
    try:
        guild = bot.get_guild(STORAGE_SERVER_ID)
        if not guild:
            log.error(f"Storage server {STORAGE_SERVER_ID} not found")
            return None
        channel = guild.get_channel(STORAGE_VARIABLES_CHANNEL_ID)
        if not channel:
            log.error(f"Variables channel {STORAGE_VARIABLES_CHANNEL_ID} not found")
        return channel
    except Exception as e:
        log.error(f"Error getting variables channel: {e}")
        return None

async def load_remote_config() -> Dict[str, Any]:
    """Load configuration from remote Discord storage"""
    global remote_storage_message_id
    try:
        channel = await get_remote_storage_channel()
        if not channel:
            return {}
        
        async for message in channel.history(limit=1):
            remote_storage_message_id = message.id
            if message.content:
                try:
                    return json.loads(message.content)
                except json.JSONDecodeError:
                    log.error("Failed to decode remote config JSON")
                    return {}
            elif message.attachments:
                # Load from attachment
                attachment = message.attachments[0]
                content = await attachment.read()
                try:
                    return json.loads(content.decode('utf-8'))
                except json.JSONDecodeError:
                    log.error("Failed to decode remote config attachment")
                    return {}
        return {}
    except Exception as e:
        log.error(f"Error loading remote config: {e}")
        return {}

async def save_remote_config(config: Dict[str, Any]) -> bool:
    """Save configuration to remote Discord storage"""
    global remote_storage_message_id
    try:
        channel = await get_remote_storage_channel()
        if not channel:
            return False
        
        json_str = json.dumps(config, indent=2)
        
        # If JSON is too large (>2000 chars), use attachment
        if len(json_str) > 2000:
            json_bytes = json_str.encode('utf-8')
            file = discord.File(BytesIO(json_bytes), filename="data.json")
            
            if remote_storage_message_id:
                try:
                    msg = await channel.fetch_message(remote_storage_message_id)
                    await msg.edit(content="", attachments=[file])
                except:
                    msg = await channel.send(file=file)
                    remote_storage_message_id = msg.id
            else:
                msg = await channel.send(file=file)
                remote_storage_message_id = msg.id
        else:
            if remote_storage_message_id:
                try:
                    msg = await channel.fetch_message(remote_storage_message_id)
                    await msg.edit(content=f"```json\n{json_str}\n```", attachments=[])
                except:
                    msg = await channel.send(f"```json\n{json_str}\n```")
                    remote_storage_message_id = msg.id
            else:
                msg = await channel.send(f"```json\n{json_str}\n```")
                remote_storage_message_id = msg.id
        
        return True
    except Exception as e:
        log.error(f"Error saving remote config: {e}")
        return False

async def load_remote_variables() -> Dict[str, Any]:
    """Load variables from remote Discord storage"""
    global remote_variables_message_id
    try:
        channel = await get_remote_variables_channel()
        if not channel:
            return {}
        
        async for message in channel.history(limit=1):
            remote_variables_message_id = message.id
            if message.content:
                try:
                    return json.loads(message.content)
                except json.JSONDecodeError:
                    return {}
        return {}
    except Exception as e:
        log.error(f"Error loading remote variables: {e}")
        return {}

async def save_remote_variables(variables: Dict[str, Any]) -> bool:
    """Save variables to remote Discord storage"""
    global remote_variables_message_id
    try:
        channel = await get_remote_variables_channel()
        if not channel:
            return False
        
        json_str = json.dumps(variables, indent=2)
        
        if remote_variables_message_id:
            try:
                msg = await channel.fetch_message(remote_variables_message_id)
                await msg.edit(content=f"```json\n{json_str}\n```")
            except:
                msg = await channel.send(f"```json\n{json_str}\n```")
                remote_variables_message_id = msg.id
        else:
            msg = await channel.send(f"```json\n{json_str}\n```")
            remote_variables_message_id = msg.id
        
        return True
    except Exception as e:
        log.error(f"Error saving remote variables: {e}")
        return False

async def log_action(action: str):
    """Log an action to the Discord logging channel"""
    try:
        channel = bot.get_channel(LOG_CHANNEL_ID)
        if channel:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await channel.send(f"```[{timestamp}] {action}```")
    except Exception as e:
        log.error(f"Failed to log action: {e}")

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

dynamic_commands_cache: Dict[str, Dict[str, Any]] = {}
guild_cache: Dict[str, Dict[str, Any]] = {}
user_chat_history: Dict[int, List[Dict]] = {}
user_codex_history: Dict[int, List[Dict]] = {}
active_hangman_games: Dict[int, Dict] = {}
active_tictactoe_games: Dict[int, Dict] = {}
active_rpg_games: Dict[int, Dict] = {}
active_word_scramble: Dict[int, Dict] = {}
locked_channels: Dict[int, Dict[str, Any]] = {}
guild_log_settings: Dict[str, Dict[str, int]] = {}
user_xp_cache: Dict[str, Dict[str, int]] = {}

EMOJI_LOCK = "<:QubeLock:1445405041371385926>"
EMOJI_CODE = "<:QubeCode:1445405028637474897>"
EMOJI_THINK = "<:QubeLink:1445405032865333248>"
EMOJI_BIN = "<:QubeBin:1445405026280538122>"
EMOJI_BELL = "<:QubeBell:1445411813796872322>"
EMOJI_CLOCK = "<:QubeClock:1445414524546056272>"
EMOJI_HAMMER = "<:QubeHammer:1445411008125472900>"
EMOJI_GAME = "<:QubeGame:1445405031036882954>"
EMOJI_APPROVED = "<:QubeTick:1445411815646560337>"
EMOJI_DENIED = "<:QubeCross:1445411932709584916>"
EMOJI_MESSAGE = "<:QubeMessage:1445413139884802149>"
EMOJI_CHECK = "<:QubeTick:1445411815646560337>"
EMOJI_EXCL = "<:QubeExclamationMark:1445414166474133515>"

HOLIDAYS = {
    (12, 25): ("Christmas", "🎄", "#c41e3a"),
    (12, 31): ("New Year's Eve", "🎆", "#ffd700"),
    (1, 1): ("New Year", "🎉", "#ffd700"),
    (10, 31): ("Halloween", "🎃", "#ff6600"),
    (2, 14): ("Valentine's Day", "💕", "#ff69b4"),
    (7, 4): ("Independence Day", "🎆", "#0000ff"),
    (19, 15): ("Qube IA birthday", "🤖", "#3D0075"),
    (11, 28): ("Thanksgiving", "🦃", "#d2691e"),
    (3, 17): ("St. Patrick's Day", "☘️", "#00ff00"),
    (4, 1): ("April Fools", "🃏", "#ff00ff"),
    (5, 5): ("Cinco de Mayo", "🌮", "#00ff00"),
}

def load_config() -> Dict[str, Any]:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def _write_config(config: Dict[str, Any]) -> bool:
    global xp_cache_dirty
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        if user_xp_cache:
            config["user_xp"] = user_xp_cache
            xp_cache_dirty = False
        with open(DATA_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        log.error(f"Failed to write data file: {e}")
        return False

def save_config(config: Dict[str, Any], merge_xp_cache: bool = True) -> bool:
    if merge_xp_cache and user_xp_cache:
        config["user_xp"] = user_xp_cache
    # Save to local file as backup
    _write_config(config)
    # Save to remote storage
    asyncio.create_task(save_remote_config(config))
    return True

async def get_guild_config(guild_id: str) -> Dict[str, Any]:
    """Get configuration for a guild - always returns a valid config dict"""
    config = load_config()
    if "guilds" not in config:
        config["guilds"] = {}
    if guild_id not in config["guilds"]:
        config["guilds"][guild_id] = {"initialized": True}
        _write_config(config)
    return config["guilds"].get(guild_id, {"initialized": True})

async def atomic_config_update(update_fn, guild_id: str) -> bool:
    async with data_lock:
        config = load_config()
        if "guilds" not in config:
            config["guilds"] = {}
        if guild_id not in config["guilds"]:
            config["guilds"][guild_id] = {"initialized": True}
        update_fn(config)
        return _write_config(config)

def get_guild_level_config(guild_id: str) -> Dict[str, Any]:
    config = load_config()
    level_configs = config.get("level_configs") or {}
    return level_configs.get(guild_id, {
        "xp_per_message": 10,
        "level_xp": 100,
        "ignored_channels": [],
        "ignored_roles": [],
        "level_roles": {}
    })

def save_guild_level_config(guild_id: str, level_config: Dict[str, Any]) -> bool:
    config = load_config()
    if "level_configs" not in config:
        config["level_configs"] = {}
    config["level_configs"][guild_id] = level_config
    if user_xp_cache:
        config["user_xp"] = user_xp_cache
    return _write_config(config)

xp_cache_dirty = False

def load_xp_cache():
    global user_xp_cache
    config = load_config()
    user_xp_cache = config.get("user_xp") or {}

def get_user_xp(guild_id: str, user_id: int) -> int:
    guild_xp = user_xp_cache.get(str(guild_id)) or {}
    return guild_xp.get(str(user_id), 0)

def save_user_xp(guild_id: str, user_id: int, xp: int) -> bool:
    global xp_cache_dirty
    if str(guild_id) not in user_xp_cache:
        user_xp_cache[str(guild_id)] = {}
    user_xp_cache[str(guild_id)][str(user_id)] = xp
    xp_cache_dirty = True
    return True

def flush_xp_cache() -> bool:
    global xp_cache_dirty
    if not xp_cache_dirty:
        return True
    config = load_config()
    config["user_xp"] = user_xp_cache
    result = _write_config(config)
    xp_cache_dirty = False
    return result

def get_user_level(guild_id: str, user_id: int) -> int:
    xp = get_user_xp(guild_id, user_id)
    level_config = get_guild_level_config(guild_id)
    level_xp = level_config.get("level_xp", 100)
    return xp // level_xp if level_xp > 0 else 0

def xp_to_next_level(guild_id: str, user_id: int) -> int:
    xp = get_user_xp(guild_id, user_id)
    level_config = get_guild_level_config(guild_id)
    level_xp = level_config.get("level_xp", 100)
    if level_xp <= 0:
        return 0
    current_level_xp = xp % level_xp
    return level_xp - current_level_xp

async def save_dynamic_command(guild_id: str, cmd_name: str, code: str, description: str = None) -> bool:
    if guild_id not in dynamic_commands_cache:
        dynamic_commands_cache[guild_id] = {}
    dynamic_commands_cache[guild_id][cmd_name] = {
        "code": code,
        "description": description or f"Dynamic command: {cmd_name}",
    }
    config = load_config()
    if "dynamic_commands" not in config:
        config["dynamic_commands"] = {}
    if guild_id not in config["dynamic_commands"]:
        config["dynamic_commands"][guild_id] = {}
    config["dynamic_commands"][guild_id][cmd_name] = {
        "code": code,
        "description": description or f"Dynamic command: {cmd_name}",
    }
    return save_config(config)

async def delete_dynamic_command_from_store(guild_id: str, cmd_name: str) -> bool:
    if guild_id in dynamic_commands_cache:
        dynamic_commands_cache[guild_id].pop(cmd_name, None)
    config = load_config()
    if "dynamic_commands" in config and guild_id in config["dynamic_commands"]:
        config["dynamic_commands"][guild_id].pop(cmd_name, None)
        return save_config(config)
    return True

class UnsafeCodeError(Exception):
    pass

ALLOWED_DUNDER_ATTRS = {
    "__name__", "__class__", "__doc__", "__module__", "__qualname__",
    "__dict__", "__slots__", "__annotations__", "__bases__", "__mro__",
    "__init__", "__new__", "__call__", "__str__", "__repr__", "__len__",
    "__iter__", "__next__", "__getitem__", "__setitem__", "__delitem__",
    "__contains__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__hash__", "__bool__", "__add__", "__sub__", "__mul__", "__truediv__",
    "__floordiv__", "__mod__", "__pow__", "__and__", "__or__", "__xor__",
    "__enter__", "__exit__", "__aenter__", "__aexit__", "__await__",
    "__aiter__", "__anext__", "__version__", "__file__", "__all__",
}

BLOCKED_DUNDER_ATTRS = {
    "__code__", "__globals__", "__builtins__", "__subclasses__",
    "__bases__", "__mro__", "__reduce__", "__reduce_ex__",
    "__getattribute__", "__setattr__", "__delattr__",
}

class SimpleASTValidator(ast.NodeVisitor):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "__import__", "compile", "execfile", "open"):
            raise UnsafeCodeError(f"Call to {node.func.id} not allowed.")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            if node.attr in BLOCKED_DUNDER_ATTRS:
                raise UnsafeCodeError(f"Access to {node.attr} not allowed for security reasons.")
        self.generic_visit(node)

def validate_user_code(code: str):
    try:
        tree = ast.parse(textwrap.dedent(code))
    except Exception as e:
        raise UnsafeCodeError(f"Invalid Python syntax: {e}")
    SimpleASTValidator().visit(tree)
    if len(code) > 10000:
        raise UnsafeCodeError("Code too long (max 10000 characters).")
    found = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            found = True
            break
    if not found:
        raise UnsafeCodeError("You must define `async def run(interaction)` or `def run(interaction)`.")

def is_admin(interaction: discord.Interaction) -> bool:
    try:
        return interaction.user.guild_permissions.manage_guild or interaction.user.guild_permissions.administrator
    except:
        return False

def is_owner(interaction: discord.Interaction) -> bool:
    try:
        return interaction.user.id == interaction.guild.owner_id
    except:
        return False

async def run_blocking(func, *args, timeout=60, **kwargs):
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(loop.run_in_executor(None, lambda: func(*args, **kwargs)), timeout=timeout)

async def ai_chat(prompt: str, system_message: str = None) -> str:
    if not gemini_client:
        return "AI not configured. Please set GEMINI_API_KEY."

    try:
        if system_message:
            full_prompt = f"{system_message}\n\nUser: {prompt}"
        else:
            full_prompt = prompt

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=2000,
                    temperature=0.7,
                )
            )
        )
        return response.text if response.text else "No response generated."
    except Exception as e:
        log.exception("AI chat error")
        return f"AI error: {str(e)}"

async def ai_generate_code(description: str) -> tuple[Optional[str], Optional[str], str]:
    if not gemini_client:
        return None, None, "AI not configured. Please set GEMINI_API_KEY."

    system_prompt = textwrap.dedent("""
    You are a Discord bot command generator. Generate Python code for Discord slash commands.

    Rules:
    1. Code MUST define a complete `async def run(interaction)` function
    2. Use interaction.response.send_message() to respond
    3. You can use discord.Embed, discord.ui.Button, discord.ui.Select for rich interactions
    4. Access bot via the global `bot` variable
    5. Access discord module via global `discord` variable
    6. Import statements ARE allowed
    7. Keep code safe and simple
    8. For buttons/selects, use discord.ui.View
    9. Do NOT use modals or text inputs
    10. ALWAYS include the COMPLETE function

    Return format:
    COMMAND_NAME: <command_name>
    DESCRIPTION: <short description>
    CODE:
    ```python
    async def run(interaction):
        await interaction.response.send_message("...")
    ```
    """)

    try:
        full_prompt = f"{system_prompt}\n\nGenerate a Discord bot command for: {description}"
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=3000,
                    temperature=0.7,
                )
            )
        )
        response_text = response.text

        cmd_name_match = re.search(r'COMMAND_NAME:\s*(\w+)', response_text, re.IGNORECASE)
        cmd_name = cmd_name_match.group(1) if cmd_name_match else None

        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?:\n|CODE:)', response_text, re.IGNORECASE | re.DOTALL)
        cmd_desc = desc_match.group(1).strip() if desc_match else "AI-generated command"

        code_match = re.search(r'```python\s*(.*?)\s*```', response_text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = None

        if not cmd_name:
            words = description.lower().split()[:2]
            cmd_name = '_'.join(w for w in words if w.isalnum())[:20]
            if not cmd_name:
                cmd_name = "custom_cmd"

        if not code or 'async def run' not in code and 'def run' not in code:
            return None, None, "Failed to extract valid code from AI response."

        return cmd_name, code, cmd_desc
    except Exception as e:
        log.exception("AI generate code error")
        return None, None, f"AI error: {str(e)}"

COMMAND_GEN_SYSTEM_PROMPT = textwrap.dedent("""
You are a Discord bot command generator. Generate Python code for Discord slash commands.

Rules:
1. Code MUST define a complete `async def run(interaction)` function
2. Use interaction.response.send_message() to respond
3. You can use discord.Embed, discord.ui.Button, discord.ui.Select for rich interactions
4. Access bot via the global `bot` variable
5. Access discord module via global `discord` variable
6. Import statements ARE allowed
7. Keep code safe and simple
8. For buttons/selects, use discord.ui.View
9. Do NOT use modals or text inputs
10. ALWAYS include the COMPLETE function

Return format:
COMMAND_NAME: <command_name>
DESCRIPTION: <short description>
CODE:
```python
async def run(interaction):
    await interaction.response.send_message("...")
```
""")

async def ai_generate_code_with_key(description: str, api_key: str, provider: str = "gemini") -> tuple[Optional[str], Optional[str], str]:
    try:
        full_prompt = f"{COMMAND_GEN_SYSTEM_PROMPT}\n\nGenerate a Discord bot command for: {description}"

        if provider.lower() == "openai":
            client = openai.OpenAI(api_key=api_key)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": COMMAND_GEN_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Generate a Discord bot command for: {description}"}
                    ],
                    max_tokens=3000,
                    temperature=0.7,
                )
            )
            response_text = response.choices[0].message.content
        else:
            temp_client = genai.Client(api_key=api_key)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: temp_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=3000,
                        temperature=0.7,
                    )
                )
            )
            response_text = response.text

        cmd_name_match = re.search(r'COMMAND_NAME:\s*(\w+)', response_text, re.IGNORECASE)
        cmd_name = cmd_name_match.group(1) if cmd_name_match else None

        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?:\n|CODE:)', response_text, re.IGNORECASE | re.DOTALL)
        cmd_desc = desc_match.group(1).strip() if desc_match else "AI-generated command"

        code_match = re.search(r'```python\s*(.*?)\s*```', response_text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = None

        if not cmd_name:
            words = description.lower().split()[:2]
            cmd_name = '_'.join(w for w in words if w.isalnum())[:20]
            if not cmd_name:
                cmd_name = "custom_cmd"

        if not code or 'async def run' not in code and 'def run' not in code:
            return None, None, "Failed to extract valid code from AI response."

        return cmd_name, code, cmd_desc
    except Exception as e:
        log.exception("AI generate code with key error")
        return None, None, f"AI error: {str(e)}"

async def register_dynamic_command(guild_id: str, cmd_name: str, code: str, description: str = None):
    try:
        validate_user_code(code)
    except UnsafeCodeError as e:
        return False, str(e)

    namespace = {"discord": discord, "bot": bot, "asyncio": asyncio, "interaction": None}

    try:
        exec(textwrap.dedent(code), namespace)
    except Exception as e:
        return False, f"Execution error: {traceback.format_exc()}"

    run_func = namespace.get("run")
    if not run_func or not callable(run_func):
        return False, "No callable `run` function found."

    guild_obj = bot.get_guild(int(guild_id))
    if not guild_obj:
        return False, f"Guild {guild_id} not found."

    async def command_wrapper(interaction: discord.Interaction):
        try:
            if inspect.iscoroutinefunction(run_func):
                await run_func(interaction)
            else:
                await run_blocking(run_func, interaction)
        except Exception as e:
            log.exception(f"Error in dynamic command {cmd_name}")
            try:
                await interaction.response.send_message(f"Command error: {str(e)}", ephemeral=True)
            except:
                try:
                    await interaction.followup.send(f"Command error: {str(e)}", ephemeral=True)
                except:
                    pass

    cmd = app_commands.Command(
        name=cmd_name,
        description=description or f"Dynamic command: {cmd_name}",
        callback=command_wrapper,
    )
    bot.tree.add_command(cmd, guild=guild_obj)
    return True, None

async def unregister_dynamic_command(guild_id: str, cmd_name: str) -> bool:
    """Remove a dynamic command from the bot's command tree"""
    try:
        guild_obj = bot.get_guild(int(guild_id))
        if not guild_obj:
            log.warning(f"Guild {guild_id} not found for command removal")
            return False

        bot.tree.remove_command(cmd_name, guild=guild_obj)
        log.info(f"Removed command '{cmd_name}' from guild {guild_id}")
        return True
    except Exception as e:
        log.exception(f"Failed to remove command '{cmd_name}' from guild {guild_id}")
        return False

async def sync_guild_commands(guild_id: str) -> bool:
    """Sync commands for a specific guild"""
    try:
        guild_obj = bot.get_guild(int(guild_id))
        if guild_obj:
            await bot.tree.sync(guild=guild_obj)
            log.info(f"Synced commands for guild {guild_id}")
            return True
        return False
    except Exception as e:
        log.exception(f"Failed to sync guild {guild_id}")
        return False

async def sync_all_guild_commands():
    for guild_id in dynamic_commands_cache.keys():
        await sync_guild_commands(guild_id)

def create_hangman_image(word: str, guessed: set, wrong_guesses: int) -> BytesIO:
    width, height = 400, 300
    img = Image.new('RGB', (width, height), color='#2C2F33')
    draw = ImageDraw.Draw(img)

    draw.line([(50, 250), (150, 250)], fill='white', width=3)
    draw.line([(100, 250), (100, 50)], fill='white', width=3)
    draw.line([(100, 50), (200, 50)], fill='white', width=3)
    draw.line([(200, 50), (200, 80)], fill='white', width=3)

    if wrong_guesses >= 1:
        draw.ellipse([(180, 80), (220, 120)], outline='white', width=3)
    if wrong_guesses >= 2:
        draw.line([(200, 120), (200, 180)], fill='white', width=3)
    if wrong_guesses >= 3:
        draw.line([(200, 140), (170, 160)], fill='white', width=3)
    if wrong_guesses >= 4:
        draw.line([(200, 140), (230, 160)], fill='white', width=3)
    if wrong_guesses >= 5:
        draw.line([(200, 180), (170, 220)], fill='white', width=3)
    if wrong_guesses >= 6:
        draw.line([(200, 180), (230, 220)], fill='white', width=3)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    display_word = ' '.join([c if c.lower() in guessed else '_' for c in word])
    draw.text((250, 150), display_word, fill='#7289DA', font=font)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

def create_tictactoe_board(board: list, player_symbol: str) -> str:
    symbols = {0: "⬜", 1: "❌", 2: "⭕"}
    lines = []
    for i in range(3):
        row = ""
        for j in range(3):
            row += symbols[board[i * 3 + j]]
        lines.append(row)
    return "\n".join(lines)

def check_tictactoe_winner(board: list) -> int:
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for line in lines:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]
    if 0 not in board:
        return -1
    return 0

def tictactoe_ai_easy(board: list) -> int:
    """Easy: Random move"""
    empty = [i for i in range(9) if board[i] == 0]
    return random.choice(empty) if empty else -1

def tictactoe_ai_medium(board: list) -> int:
    """Medium: Win/Block strategy"""
    for i in range(9):
        if board[i] == 0:
            board[i] = 2
            if check_tictactoe_winner(board) == 2:
                board[i] = 0
                return i
            board[i] = 0

    for i in range(9):
        if board[i] == 0:
            board[i] = 1
            if check_tictactoe_winner(board) == 1:
                board[i] = 0
                return i
            board[i] = 0

    empty = [i for i in range(9) if board[i] == 0]
    return random.choice(empty) if empty else -1

def tictactoe_ai_hard(board: list) -> int:
    """Hard: Optimal play"""
    for i in range(9):
        if board[i] == 0:
            board[i] = 2
            if check_tictactoe_winner(board) == 2:
                board[i] = 0
                return i
            board[i] = 0

    for i in range(9):
        if board[i] == 0:
            board[i] = 1
            if check_tictactoe_winner(board) == 1:
                board[i] = 0
                return i
            board[i] = 0

    if board[4] == 0:
        return 4

    corners = [0, 2, 6, 8]
    random.shuffle(corners)
    for c in corners:
        if board[c] == 0:
            return c

    edges = [1, 3, 5, 7]
    random.shuffle(edges)
    for e in edges:
        if board[e] == 0:
            return e

    return -1

def tictactoe_ai_move(board: list, difficulty: str = "hard") -> int:
    """Wrapper for AI move with difficulty level"""
    if difficulty == "easy":
        return tictactoe_ai_easy(board)
    elif difficulty == "medium":
        return tictactoe_ai_medium(board)
    else:
        return tictactoe_ai_hard(board)

async def create_welcome_image(member: discord.Member, is_join: bool = True) -> BytesIO:
    """Create a beautiful welcome/goodbye image using PIL"""
    width, height = 800, 300

    # Create gradient background
    base_color = "#4CAF50" if is_join else "#F44336"
    img = Image.new('RGB', (width, height), color=base_color)
    draw = ImageDraw.Draw(img)

    # Gradient effect
    for i in range(height):
        shade_factor = 1 - (i / height) * 0.4
        r = int(int(base_color[1:3], 16) * shade_factor)
        g = int(int(base_color[3:5], 16) * shade_factor)
        b = int(int(base_color[5:7], 16) * shade_factor)
        draw.rectangle(((0, i), (width, i + 1)), fill=f'#{r:02x}{g:02x}{b:02x}')

    # Load fonts
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()

    # Draw title
    title = f"Welcome, {member.display_name}!" if is_join else f"Goodbye, {member.display_name}!"
    bbox = draw.textbbox((0, 0), title, font=font_large)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, 80), title, fill='white', font=font_large)

    # Draw subtitle
    subtitle = "Just joined the server!" if is_join else "Left the server"
    bbox = draw.textbbox((0, 0), subtitle, font=font_medium)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, 160), subtitle, fill='white', font=font_medium)

    # Draw decorative border
    draw.rectangle(((10, 10), (width - 10, height - 10)), outline='white', width=5)

    # Add emoji decorations
    emoji = "👋" if is_join else "😢"
    draw.text((50, 220), emoji, fill='white', font=font_large)
    draw.text((width - 100, 220), emoji, fill='white', font=font_large)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

def create_holiday_postcard(holiday_name: str, emoji: str, color: str) -> BytesIO:
    width, height = 800, 400
    img = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(img)

    for i in range(height):
        shade_factor = 1 - (i / height) * 0.3
        r = int(int(color[1:3], 16) * shade_factor)
        g = int(int(color[3:5], 16) * shade_factor)
        b = int(int(color[5:7], 16) * shade_factor)
        draw.rectangle(((0, i), (width, i + 1)), fill=f'#{r:02x}{g:02x}{b:02x}')

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        font_emoji = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 64)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_emoji = ImageFont.load_default()

    title = f"Happy {holiday_name}!"
    bbox = draw.textbbox((0, 0), title, font=font_large)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, 100), title, fill='white', font=font_large)

    message = "Wishing you joy and happiness!"
    bbox = draw.textbbox((0, 0), message, font=font_medium)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, 200), message, fill='white', font=font_medium)

    draw.text((50, 280), emoji, fill='white', font=font_emoji)
    draw.text((width - 100, 280), emoji, fill='white', font=font_emoji)

    draw.rectangle(((10, 10), (width - 10, height - 10)), outline='white', width=3)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

class HangmanLetterSelect(Select):
    def __init__(self, user_id: int, guessed: set):
        options = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if letter.lower() not in guessed:
                options.append(discord.SelectOption(label=letter, value=letter.lower()))
        if len(options) > 25:
            options = options[:25]
        super().__init__(placeholder="Select a letter...", options=options)
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("This is not your game!", ephemeral=True)

        game = active_hangman_games.get(self.user_id)
        if not game:
            return await interaction.response.send_message("No active game found!", ephemeral=True)

        letter = self.values[0]
        game["guessed"].add(letter)

        if letter not in game["word"].lower():
            game["wrong"] += 1

        word = game["word"]
        guessed = game["guessed"]
        wrong = game["wrong"]

        if all(c.lower() in guessed for c in word):
            del active_hangman_games[self.user_id]
            image_buffer = create_hangman_image(word, guessed, wrong)
            file = discord.File(image_buffer, filename="hangman.png")
            embed = discord.Embed(
                title="🎉 You Won!",
                description=f"The word was: **{word}**",
                color=discord.Color.green()
            )
            embed.set_image(url="attachment://hangman.png")
            return await interaction.response.edit_message(embed=embed, attachments=[file], view=None)

        if wrong >= 6:
            del active_hangman_games[self.user_id]
            image_buffer = create_hangman_image(word, guessed, wrong)
            file = discord.File(image_buffer, filename="hangman.png")
            embed = discord.Embed(
                title="💀 Game Over!",
                description=f"The word was: **{word}**",
                color=discord.Color.red()
            )
            embed.set_image(url="attachment://hangman.png")
            return await interaction.response.edit_message(embed=embed, attachments=[file], view=None)

        image_buffer = create_hangman_image(word, guessed, wrong)
        file = discord.File(image_buffer, filename="hangman.png")
        embed = discord.Embed(
            title="🎯 Hangman",
            description=f"Wrong guesses: {wrong}/6\nGuessed: {', '.join(sorted(guessed))}",
            color=discord.Color.blue()
        )
        embed.set_image(url="attachment://hangman.png")

        view = View(timeout=300)
        view.add_item(HangmanLetterSelect(self.user_id, guessed))
        await interaction.response.edit_message(embed=embed, attachments=[file], view=view)

class TicTacToeButton(Button):
    def __init__(self, position: int, user_id: int, is_singleplayer: bool, difficulty: str = "hard"):
        super().__init__(style=discord.ButtonStyle.secondary, label="⬜", row=position // 3)
        self.position = position
        self.user_id = user_id
        self.is_singleplayer = is_singleplayer
        self.difficulty = difficulty

    async def callback(self, interaction: discord.Interaction):
        game = active_tictactoe_games.get(self.user_id)
        if not game:
            return await interaction.response.send_message("No active game!", ephemeral=True)

        if self.is_singleplayer:
            if interaction.user.id != self.user_id:
                return await interaction.response.send_message("This is not your game!", ephemeral=True)
        else:
            current_player = game["current_player"]
            if current_player == 1 and interaction.user.id != game["player1"]:
                return await interaction.response.send_message("It's not your turn!", ephemeral=True)
            if current_player == 2 and interaction.user.id != game["player2"]:
                return await interaction.response.send_message("It's not your turn!", ephemeral=True)

        if game["board"][self.position] != 0:
            return await interaction.response.send_message("That cell is taken!", ephemeral=True)

        current = game["current_player"]
        game["board"][self.position] = current

        self.label = "❌" if current == 1 else "⭕"
        self.style = discord.ButtonStyle.danger if current == 1 else discord.ButtonStyle.primary
        self.disabled = True

        winner = check_tictactoe_winner(game["board"])

        if winner != 0:
            for item in self.view.children:
                if isinstance(item, Button):
                    item.disabled = True

            if winner == -1:
                result = "It's a tie! 🤝"
            elif self.is_singleplayer:
                if winner == 1:
                    result = "You won! 🎉"
                    asyncio.create_task(log_action(f"User {self.user_id} won Tic-Tac-Toe vs AI ({game.get('difficulty', 'hard')})"))
                else:
                    result = "AI wins! 🤖"
                    asyncio.create_task(log_action(f"User {self.user_id} lost Tic-Tac-Toe vs AI ({game.get('difficulty', 'hard')})"))
            elif winner == -1:
                asyncio.create_task(log_action(f"User {self.user_id} drew Tic-Tac-Toe multiplayer"))
            else:
                winner_name = "Player 1 (❌)" if winner == 1 else "Player 2 (⭕)"
                result = f"{winner_name} wins! 🎉"

            del active_tictactoe_games[self.user_id]
            embed = discord.Embed(title="Tic-Tac-Toe", description=result, color=discord.Color.gold())
            return await interaction.response.edit_message(embed=embed, view=self.view)

        if self.is_singleplayer and current == 1:
            ai_pos = tictactoe_ai_move(game["board"], game.get("difficulty", "hard"))
            if ai_pos != -1:
                game["board"][ai_pos] = 2
                for item in self.view.children:
                    if isinstance(item, TicTacToeButton) and item.position == ai_pos:
                        item.label = "⭕"
                        item.style = discord.ButtonStyle.primary
                        item.disabled = True

                winner = check_tictactoe_winner(game["board"])
                if winner != 0:
                    for item in self.view.children:
                        if isinstance(item, Button):
                            item.disabled = True

                    if winner == -1:
                        result = "It's a tie! 🤝"
                    else:
                        result = "AI wins! 🤖"

                    del active_tictactoe_games[self.user_id]
                    embed = discord.Embed(title="Tic-Tac-Toe", description=result, color=discord.Color.gold())
                    return await interaction.response.edit_message(embed=embed, view=self.view)
        else:
            game["current_player"] = 2 if current == 1 else 1

        current_symbol = "❌" if game["current_player"] == 1 else "⭕"
        embed = discord.Embed(
            title="Tic-Tac-Toe",
            description=f"Current turn: {current_symbol}",
            color=discord.Color.blue()
        )
        await interaction.response.edit_message(embed=embed, view=self.view)

class RPGChoiceButton(Button):
    def __init__(self, choice: str, user_id: int):
        super().__init__(style=discord.ButtonStyle.primary, label=choice)
        self.choice = choice
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("This is not your adventure!", ephemeral=True)

        game = active_rpg_games.get(self.user_id)
        if not game:
            return await interaction.response.send_message("No active adventure!", ephemeral=True)

        scenario = game["scenario"]
        outcome = scenario["outcomes"][self.choice]

        success = random.random() < outcome["success_chance"]
        result = outcome["success"] if success else outcome["fail"]

        color = discord.Color.green() if success else discord.Color.red()
        emoji = "✅" if success else "❌"

        embed = discord.Embed(
            title=f"{emoji} {self.choice}!",
            description=result,
            color=color
        )

        del active_rpg_games[self.user_id]

        view = View(timeout=60)
        play_again = Button(label="🎮 Play Again", style=discord.ButtonStyle.success)

        async def play_again_callback(btn_interaction: discord.Interaction):
            if btn_interaction.user.id != self.user_id:
                return await btn_interaction.response.send_message("Start your own adventure with /rpg!", ephemeral=True)

            scenario = random.choice(RPG_SCENARIOS_LIST)
            active_rpg_games[self.user_id] = {"scenario": scenario}

            embed = discord.Embed(
                title="⚔️ RPG Mini Adventure",
                description=scenario["intro"],
                color=discord.Color.purple()
            )
            embed.add_field(name="What do you do?", value="Choose wisely...", inline=False)

            new_view = View(timeout=300)
            for option in scenario["options"]:
                new_view.add_item(RPGChoiceButton(option, self.user_id))

            await btn_interaction.response.edit_message(embed=embed, view=new_view)

        play_again.callback = play_again_callback
        view.add_item(play_again)

        await interaction.response.edit_message(embed=embed, view=view)

class EjectIntruderButton(Button):
    def __init__(self, emoji: str, is_intruder: bool, user_id: int):
        super().__init__(style=discord.ButtonStyle.secondary, label=emoji)
        self.is_intruder = is_intruder
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("This is not your game!", ephemeral=True)

        for item in self.view.children:
            if isinstance(item, Button):
                item.disabled = True

        if self.is_intruder:
            embed = discord.Embed(
                title="🎉 Correct!",
                description=f"You found the intruder: {self.label}",
                color=discord.Color.green()
            )
        else:
            embed = discord.Embed(
                title="❌ Wrong!",
                description=f"That wasn't the intruder!",
                color=discord.Color.red()
            )

        await interaction.response.edit_message(embed=embed, view=self.view)

class WordScrambleModal(Modal):
    answer = TextInput(label="Your Answer", placeholder="Type the unscrambled word...", required=True, max_length=50)

    def __init__(self, user_id: int):
        super().__init__(title="Word Scramble")
        self.user_id = user_id

    async def on_submit(self, interaction: discord.Interaction):
        game = active_word_scramble.get(self.user_id)
        if not game:
            return await interaction.response.send_message("No active game!", ephemeral=True)

        if self.answer.value.lower().strip() == game["word"].lower():
            embed = discord.Embed(
                title="🎉 Correct!",
                description=f"The word was: **{game['word']}**",
                color=discord.Color.green()
            )
            del active_word_scramble[self.user_id]
        else:
            embed = discord.Embed(
                title="❌ Wrong!",
                description=f"Try again! The scrambled word is: **{game['scrambled']}**",
                color=discord.Color.red()
            )

        await interaction.response.send_message(embed=embed, ephemeral=True)

class ChatContinueModal(Modal):
    message_input = TextInput(
        label="Your Message",
        style=discord.TextStyle.paragraph,
        placeholder="Type your message here...",
        required=True,
        max_length=500
    )

    def __init__(self, is_codex: bool = False):
        super().__init__(title="Continue Chat")
        self.is_codex = is_codex

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)

        user_message = self.message_input.value
        history = (user_codex_history if self.is_codex else user_chat_history).get(interaction.user.id, [])
        history.append({"role": "user", "content": user_message})

        if self.is_codex:
            system_msg = "You are Codex, an expert coding assistant. Provide clear, concise code examples and explanations. Focus on best practices and efficiency."
        else:
            system_msg = "You are a helpful, friendly assistant. Be concise but thorough in your responses."

        response = await ai_chat(user_message, system_msg)

        history.append({"role": "assistant", "content": response})
        if self.is_codex:
            user_codex_history[interaction.user.id] = history[-20:]
        else:
            user_chat_history[interaction.user.id] = history[-20:]

        view = View(timeout=300)
        continue_btn = Button(label="Continue Chat", style=discord.ButtonStyle.primary)

        async def continue_callback(btn_interaction: discord.Interaction):
            await btn_interaction.response.send_modal(ChatContinueModal(is_codex=self.is_codex))

        continue_btn.callback = continue_callback
        view.add_item(continue_btn)

        emoji = "💻" if self.is_codex else "💬"
        await interaction.followup.send(f"{emoji} **Response:**\n{response[:1900]}", view=view, ephemeral=True)

@bot.event
async def on_ready():
    log.info(f"Bot ready as {bot.user}")

    # Start background tasks immediately
    if not holiday_check.is_running():
        holiday_check.start()
    if not haunted_message_check.is_running():
        haunted_message_check.start()
    if not check_channel_unlocks.is_running():
        check_channel_unlocks.start()
    if not xp_cache_flush.is_running():
        xp_cache_flush.start()
    if not daily_trivia_check.is_running():
        daily_trivia_check.start()
    if not easter_egg_check.is_running():
        easter_egg_check.start()

    # Defer heavy operations to background task
    asyncio.create_task(initialize_bot_data())

async def initialize_bot_data():
    """Background initialization to not block bot startup"""
    try:
        # Load remote config first
        remote_config = await load_remote_config()
        if remote_config:
            # Merge with local config
            local_config = load_config()
            local_config.update(remote_config)
            _write_config(local_config)
            log.info("Remote config loaded and merged")
        
        # Load XP cache into memory
        load_xp_cache()

        # Sync commands (this is slow)
        await sync_all_guild_commands()
        await bot.tree.sync()
        log.info("All commands synced")

        # Notify dev user after everything is ready
        try:
            dev_user = await bot.fetch_user(DEV_USER_ID)
            if dev_user:
                embed = discord.Embed(
                    title="🟢 Bot Fully Loaded!",
                    description=f"**{bot.user.name}** is fully initialized!",
                    color=discord.Color.green(),
                    timestamp=datetime.utcnow()
                )
                embed.add_field(name="Servers", value=str(len(bot.guilds)), inline=True)
                embed.add_field(name="Users", value=str(sum(g.member_count or 0 for g in bot.guilds)), inline=True)
                await dev_user.send(embed=embed)
        except Exception as e:
            log.error(f"Failed to notify dev user: {e}")

    except Exception as e:
        log.exception("Error during bot initialization")

@tasks.loop(hours=1)
async def holiday_check():
    now = datetime.now()
    today = (now.month, now.day)

    if today in HOLIDAYS and now.hour == 12:
        holiday_name, emoji, color = HOLIDAYS[today]
        log.info(f"Sending holiday messages for {holiday_name}")

        guilds = list(bot.guilds)
        random.shuffle(guilds)

        postcard_buffer = await create_holiday_postcard(holiday_name, emoji, color)

        for guild in guilds:
            try:
                public_channels = [
                    c for c in guild.text_channels 
                    if c.permissions_for(guild.default_role).send_messages 
                    and c.permissions_for(guild.me).send_messages
                ]

                if not public_channels:
                    log.info(f"No public channels in {guild.name}, skipping")
                    continue

                channel = random.choice(public_channels)

                postcard_buffer.seek(0)
                file = discord.File(postcard_buffer, filename="holiday_postcard.png")

                embed = discord.Embed(
                    title=f"{emoji} Happy {holiday_name}! {emoji}",
                    description="Wishing everyone a wonderful celebration!",
                    color=discord.Color.from_str(color)
                )
                embed.set_image(url="attachment://holiday_postcard.png")

                await channel.send(embed=embed, file=file)
                log.info(f"Sent holiday message to {guild.name} in #{channel.name}")
                await asyncio.sleep(2)
            except Exception as e:
                log.error(f"Failed to send holiday message to {guild.name}: {e}")

TRIVIA_QUESTIONS = [
    {"question": "Who invented pizza?", "answer": "italians"},
    {"question": "What is the capital of France?", "answer": "paris"},
    {"question": "Who painted the Mona Lisa?", "answer": "leonardo da vinci"},
    {"question": "What year did World War II end?", "answer": "1945"},
    {"question": "What is the largest planet in our solar system?", "answer": "jupiter"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "william shakespeare"},
    {"question": "What is the chemical symbol for gold?", "answer": "au"},
    {"question": "How many continents are there?", "answer": "7"},
]

class TriviaAnswerModal(Modal, title="Answer the Trivia"):
    answer_input = TextInput(
        label="Your Answer",
        placeholder="Type your answer here...",
        required=True,
        max_length=100
    )

    def __init__(self, correct_answer: str, question: str):
        super().__init__()
        self.correct_answer = correct_answer.lower()
        self.question = question

    async def on_submit(self, interaction: discord.Interaction):
        user_answer = self.answer_input.value.lower().strip()

        if user_answer == self.correct_answer:
            embed = discord.Embed(
                title="🎉 Correct!",
                description=f"You answered correctly: **{self.correct_answer}**",
                color=discord.Color.green()
            )
        else:
            embed = discord.Embed(
                title="❌ Wrong!",
                description=f"The correct answer was: **{self.correct_answer}**",
                color=discord.Color.red()
            )

        await interaction.response.send_message(embed=embed, ephemeral=True)

def create_easter_egg_image(is_claimed: bool = False, reward: str = "candy") -> BytesIO:
    """Create easter egg image - intact or cracked with reward"""
    width, height = 400, 400
    img = Image.new('RGB', (width, height), color='#87CEEB')
    draw = ImageDraw.Draw(img)

    # Gradient sky background
    for i in range(height):
        shade_factor = 1 - (i / height) * 0.3
        r = int(0x87 * shade_factor)
        g = int(0xCE * shade_factor)
        b = int(0xEB * shade_factor)
        draw.rectangle(((0, i), (width, i + 1)), fill=f'#{r:02x}{g:02x}{b:02x}')

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()

    if is_claimed:
        # Draw cracked egg
        draw.ellipse([(120, 150), (280, 310)], fill='#FFE4B5', outline='#8B4513', width=3)
        draw.line([(200, 150), (220, 310)], fill='#8B4513', width=3)
        draw.line([(150, 200), (250, 220)], fill='#8B4513', width=3)

        # Draw reward inside
        draw.text((140, 240), f"🍬 {reward}", fill='#FF69B4', font=font_large)

        title = "Easter Egg Claimed!"
        draw.text((80, 50), title, fill='#FFD700', font=font_large)
    else:
        # Draw intact egg with pattern
        draw.ellipse([(120, 150), (280, 310)], fill='#FF69B4', outline='#8B4513', width=3)

        # Decorative pattern
        for i in range(5):
            y = 170 + i * 30
            draw.line([(140, y), (260, y)], fill='white', width=2)

        title = "🥚 Easter Egg Found!"
        draw.text((60, 50), title, fill='#FFD700', font=font_large)
        draw.text((80, 350), "Click to claim!", fill='white', font=font_medium)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

@tasks.loop(hours=24)
async def easter_egg_check():
    """Send daily Easter eggs from March 27 to April 23"""
    now = datetime.now()

    # Check if we're in Easter egg period
    if not ((now.month == 3 and now.day >= 27) or (now.month == 4 and now.day <= 23)):
        return

    rewards = ["candy", "chocolate", "toy", "coin", "gem"]

    for guild in bot.guilds:
        try:
            # Find a channel everyone can talk in
            public_channels = [
                c for c in guild.text_channels 
                if c.permissions_for(guild.default_role).send_messages 
                and c.permissions_for(guild.me).send_messages
            ]

            if not public_channels:
                continue

            channel = random.choice(public_channels)
            reward = random.choice(rewards)

            image_buffer = create_easter_egg_image(is_claimed=False)
            file = discord.File(image_buffer, filename="easter_egg.png")

            embed = discord.Embed(
                title="🥚 Easter Egg Hunt!",
                description="An Easter egg has appeared! Click the button to claim it!",
                color=discord.Color.from_str("#FF69B4")
            )
            embed.set_image(url="attachment://easter_egg.png")

            view = View(timeout=None)
            claim_btn = Button(label="Claim Egg", style=discord.ButtonStyle.success, emoji="🥚")

            async def claim_callback(btn_interaction: discord.Interaction):
                # Create claimed egg image
                claimed_buffer = create_easter_egg_image(is_claimed=True, reward=reward)
                claimed_file = discord.File(claimed_buffer, filename="claimed_egg.png")

                claim_embed = discord.Embed(
                    title="🎉 Easter Egg Claimed!",
                    description=f"{btn_interaction.user.mention} found: **{reward}**!",
                    color=discord.Color.gold()
                )
                claim_embed.set_image(url="attachment://claimed_egg.png")

                await btn_interaction.response.send_message(embed=claim_embed, file=claimed_file)

                # Disable button
                for item in view.children:
                    item.disabled = True
                await btn_interaction.message.edit(view=view)

            claim_btn.callback = claim_callback
            view.add_item(claim_btn)

            await channel.send(embed=embed, file=file, view=view)

        except Exception as e:
            log.error(f"Failed to send Easter egg to {guild.name}: {e}")

@tasks.loop(hours=24)
async def daily_trivia_check():
    """Send daily trivia questions to servers"""
    trivia = random.choice(TRIVIA_QUESTIONS)

    for guild in bot.guilds:
        try:
            # Find a channel everyone can talk in
            public_channels = [
                c for c in guild.text_channels 
                if c.permissions_for(guild.default_role).send_messages 
                and c.permissions_for(guild.me).send_messages
            ]

            if not public_channels:
                continue

            channel = random.choice(public_channels)

            embed = discord.Embed(
                title="❓ Daily Trivia Question",
                description=trivia["question"],
                color=discord.Color.blue()
            )
            embed.set_footer(text="Click the button below to answer!")

            view = View(timeout=120)
            answer_btn = Button(label="Answer", style=discord.ButtonStyle.primary, emoji="✍️")

            async def answer_callback(btn_interaction: discord.Interaction):
                await btn_interaction.response.send_modal(TriviaAnswerModal(trivia["answer"], trivia["question"]))

            answer_btn.callback = answer_callback
            view.add_item(answer_btn)

            msg = await channel.send(embed=embed, view=view)

            # Delete after 2 minutes
            await asyncio.sleep(120)
            try:
                await msg.delete()
            except:
                pass

        except Exception as e:
            log.error(f"Failed to send trivia to {guild.name}: {e}")

@tasks.loop(hours=6)
async def haunted_message_check():
    now = datetime.now()

    if now.month == 10 and 20 <= now.day <= 30:
        guilds = list(bot.guilds)
        random.shuffle(guilds)

        for guild in guilds[:3]:
            try:
                channel = guild.system_channel or next(
                    (c for c in guild.text_channels if c.permissions_for(guild.me).send_messages),
                    None
                )
                if channel:
                    message = random.choice(HAUNTED_MESSAGES)
                    await channel.send(message)
                    log.info(f"Sent haunted message to {guild.name}")
            except Exception as e:
                log.error(f"Failed to send haunted message to {guild.name}: {e}")

@bot.tree.command(name="level-config", description="Configure leveling system (Manage Server)")
@app_commands.describe(
    xp_per_message="XP earned per message (default: 10)",
    level_xp="Total XP needed per level (default: 100)",
    ignore_channel="Channel to ignore for XP (optional)",
    ignore_role="Role to ignore for XP (optional)"
)
@app_commands.checks.has_permissions(manage_guild=True)
async def cmd_level_config(
    interaction: discord.Interaction,
    xp_per_message: int = None,
    level_xp: int = None,
    ignore_channel: discord.TextChannel = None,
    ignore_role: discord.Role = None
):
    guild_id = str(interaction.guild.id)
    level_config = get_guild_level_config(guild_id)

    if xp_per_message is not None:
        level_config["xp_per_message"] = max(1, xp_per_message)

    if level_xp is not None:
        level_config["level_xp"] = max(1, level_xp)

    if ignore_channel:
        if "ignored_channels" not in level_config:
            level_config["ignored_channels"] = []
        if ignore_channel.id not in level_config["ignored_channels"]:
            level_config["ignored_channels"].append(ignore_channel.id)

    if ignore_role:
        if "ignored_roles" not in level_config:
            level_config["ignored_roles"] = []
        if ignore_role.id not in level_config["ignored_roles"]:
            level_config["ignored_roles"].append(ignore_role.id)

    save_guild_level_config(guild_id, level_config)

    embed = discord.Embed(
        title="⚙️ Leveling Config Updated",
        description="Leveling system configuration has been saved.",
        color=discord.Color.green()
    )
    embed.add_field(name="XP Per Message", value=str(level_config.get("xp_per_message", 10)), inline=True)
    embed.add_field(name="XP Per Level", value=str(level_config.get("level_xp", 100)), inline=True)
    embed.add_field(name="Ignored Channels", value=str(len(level_config.get("ignored_channels", []))), inline=True)
    embed.add_field(name="Ignored Roles", value=str(len(level_config.get("ignored_roles", []))), inline=True)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="level-setup", description="Setup level roles (Manage Server)")
@app_commands.describe(
    level="Level number (1-10)",
    role="Role to assign at this level"
)
@app_commands.checks.has_permissions(manage_guild=True)
async def cmd_level_setup(interaction: discord.Interaction, level: int, role: discord.Role):
    if level < 1 or level > 10:
        return await interaction.response.send_message("Level must be between 1 and 10.", ephemeral=True)

    guild_id = str(interaction.guild.id)
    level_config = get_guild_level_config(guild_id)

    if "level_roles" not in level_config:
        level_config["level_roles"] = {}

    for lvl, role_id in level_config["level_roles"].items():
        if role_id == role.id and lvl != str(level):
            return await interaction.response.send_message(
                f"This role is already assigned to level {lvl}. Remove it first before assigning to a different level.",
                ephemeral=True
            )

    guild_role = interaction.guild.get_role(role.id)
    if not guild_role:
        return await interaction.response.send_message(
            "This role no longer exists in the server.",
            ephemeral=True
        )

    level_config["level_roles"][str(level)] = role.id
    save_guild_level_config(guild_id, level_config)

    embed = discord.Embed(
        title="🎯 Level Role Setup",
        description=f"Level **{level}** will reward role {role.mention}",
        color=discord.Color.green()
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="level", description="Check your current level and XP")
async def cmd_level(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    guild_id = str(interaction.guild.id)

    current_xp = get_user_xp(guild_id, target.id)
    current_level = get_user_level(guild_id, target.id)
    xp_needed = xp_to_next_level(guild_id, target.id)
    level_config = get_guild_level_config(guild_id)

    embed = discord.Embed(
        title=f"📊 Level Stats for {target.display_name}",
        color=discord.Color.blue()
    )
    embed.add_field(name="Level", value=str(current_level), inline=True)
    embed.add_field(name="Total XP", value=str(current_xp), inline=True)
    embed.add_field(name="XP to Next Level", value=str(xp_needed), inline=True)
    embed.set_thumbnail(url=target.display_avatar.url)

    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="leaderboard", description="View the server XP leaderboard")
async def cmd_leaderboard(interaction: discord.Interaction):
    guild_id = str(interaction.guild.id)
    config = load_config()
    user_xp_data = config.get("user_xp", {}).get(guild_id, {})

    if not user_xp_data:
        return await interaction.response.send_message("No XP data yet! Start chatting to earn XP.", ephemeral=True)

    sorted_users = sorted(user_xp_data.items(), key=lambda x: x[1], reverse=True)[:10]

    embed = discord.Embed(
        title="🏆 XP Leaderboard",
        color=discord.Color.gold()
    )

    leaderboard_text = ""
    medals = ["🥇", "🥈", "🥉"]
    for i, (user_id, xp) in enumerate(sorted_users):
        medal = medals[i] if i < 3 else f"**{i+1}.**"
        level = xp // get_guild_level_config(guild_id).get("level_xp", 100)
        try:
            member = interaction.guild.get_member(int(user_id))
            name = member.display_name if member else f"User {user_id}"
        except:
            name = f"User {user_id}"
        leaderboard_text += f"{medal} {name} - Level {level} ({xp} XP)\n"

    embed.description = leaderboard_text or "No users yet."
    await interaction.response.send_message(embed=embed)

def create_help_image() -> BytesIO:
    """Create a beautiful help image using PIL"""
    width, height = 1000, 800
    img = Image.new('RGB', (width, height), color='#7289DA')
    draw = ImageDraw.Draw(img)

    # Gradient background
    for i in range(height):
        shade_factor = 1 - (i / height) * 0.3
        r = int(0x72 * shade_factor)
        g = int(0x89 * shade_factor)
        b = int(0xDA * shade_factor)
        draw.rectangle(((0, i), (width, i + 1)), fill=f'#{r:02x}{g:02x}{b:02x}')

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        font_section = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font_title = ImageFont.load_default()
        font_section = ImageFont.load_default()
        font_text = ImageFont.load_default()

    # Title
    draw.text((50, 40), "🤖 Bot Help Guide", fill='white', font=font_title)

    y_pos = 130

    # Mini-Games
    draw.text((50, y_pos), "🎮 Mini-Games", fill='#FFD700', font=font_section)
    y_pos += 45
    draw.text((70, y_pos), "/hangman - Word guessing game", fill='white', font=font_text)
    y_pos += 30
    draw.text((70, y_pos), "/tictactoe - Play vs AI or friend", fill='white', font=font_text)
    y_pos += 30
    draw.text((70, y_pos), "/rpg - Mini adventure", fill='white', font=font_text)
    y_pos += 50

    # AI Features
    draw.text((50, y_pos), "🤖 AI Features", fill='#FFD700', font=font_section)
    y_pos += 45
    draw.text((70, y_pos), "/chat - Chat with AI (ephemeral)", fill='white', font=font_text)
    y_pos += 30
    draw.text((70, y_pos), "/codex - Coding help from AI", fill='white', font=font_text)
    y_pos += 50

    # Admin Tools
    draw.text((50, y_pos), "⚙️ Admin Tools", fill='#FFD700', font=font_section)
    y_pos += 45
    draw.text((70, y_pos), "/level-config - Configure XP system", fill='white', font=font_text)
    y_pos += 30
    draw.text((70, y_pos), "/log - Event logging setup", fill='white', font=font_text)

    # Border
    draw.rectangle(((15, 15), (width - 15, height - 15)), outline='white', width=5)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

@bot.tree.command(name="help", description="Get help about the bot")
@app_commands.describe(question="Your question about the bot")
async def cmd_help(interaction: discord.Interaction, question: str = None):
    await interaction.response.defer(thinking=True, ephemeral=True)

    if question and gemini_client:
        faq_system = """You are a helpful Discord bot assistant. Answer questions about the bot's features.

**FEATURES:**
- /describe_command - Generate commands from description using AI
- /create_command - Upload Python file to create custom command
- /level-config - Configure XP settings (Manage Server)
- /level-setup - Assign role rewards for levels (Manage Server)
- /send - Send anonymous messages with embeds and sticky notes

**MINI-GAMES:**
- /hangman - Classic word guessing game with images
- /tictactoe - Play vs AI or another player (easy/medium/hard)
- /rpg - Mini adventure with choices
- /eject - Find the odd emoji out
- /scramble - Unscramble words

**FUN:**
- /cat - Random cute cat picture
- /funfact - Random fun fact

**AI (Ephemeral/Private):**
- /chat - Chat with AI
- /codex - Coding help from AI

Keep responses concise and helpful."""

        response = await ai_chat(question, faq_system)
        await interaction.followup.send(f"💙 {response[:2000]}", ephemeral=True)
    else:
        try:
            image_buffer = create_help_image()
            file = discord.File(image_buffer, filename="help.png")
            await interaction.followup.send(file=file, ephemeral=True)
        except Exception as e:
            log.error(f"Failed to create help image: {e}")
            embed = discord.Embed(
                title="🤖 Bot Help",
                description="Here's what I can do!",
                color=discord.Color.blue()
            )
            embed.add_field(
                name="🎮 Mini-Games",
                value="`/hangman` `/tictactoe` `/rpg` `/eject` `/scramble`",
                inline=True
            )
            embed.add_field(
                name="🤖 AI",
                value="`/chat` `/codex`",
                inline=True
            )
            await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="hangman", description="Play Hangman - guess the word!")
async def cmd_hangman(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    await log_action(f"User {interaction.user.id} ({interaction.user.name}) started Hangman")

    word = random.choice(HANGMAN_WORDS).upper()
    active_hangman_games[interaction.user.id] = {
        "word": word,
        "guessed": set(),
        "wrong": 0
    }

    image_buffer = create_hangman_image(word, set(), 0)
    file = discord.File(image_buffer, filename="hangman.png")

    embed = discord.Embed(
        title="🎯 Hangman",
        description="Guess the word by selecting letters!",
        color=discord.Color.blue()
    )
    embed.set_image(url="attachment://hangman.png")

    view = View(timeout=300)
    view.add_item(HangmanLetterSelect(interaction.user.id, set()))

    await interaction.followup.send(embed=embed, file=file, view=view, ephemeral=True)


@bot.tree.command(name="rpg", description="Start a mini RPG adventure!")
async def cmd_rpg(interaction: discord.Interaction):
    scenario = random.choice(RPG_SCENARIOS_LIST)
    active_rpg_games[interaction.user.id] = {"scenario": scenario}

    embed = discord.Embed(
        title="⚔️ RPG Mini Adventure",
        description=scenario["intro"],
        color=discord.Color.purple()
    )
    embed.add_field(name="What do you do?", value="Choose wisely...", inline=False)

    view = View(timeout=300)
    for option in scenario["options"]:
        view.add_item(RPGChoiceButton(option, interaction.user.id))

    await log_action(f"User {interaction.user.id} ({interaction.user.name}) started RPG Adventure")
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

@bot.tree.command(name="eject", description="Find the odd emoji out!")
async def cmd_eject(interaction: discord.Interaction):
    pattern = random.choice(INTRUDER_PATTERNS_LIST)
    main_emojis = pattern["main_emojis"]
    intruder = pattern["intruder"]

    all_emojis = main_emojis + [intruder]
    random.shuffle(all_emojis)

    embed = discord.Embed(
        title="🔍 Eject the Intruder!",
        description=f"Category: **{pattern['main_category']}**\n\nFind the emoji that doesn't belong!",
        color=discord.Color.orange()
    )
    embed.add_field(name="Emojis", value=" ".join(all_emojis), inline=False)

    view = View(timeout=60)
    for emoji in all_emojis:
        view.add_item(EjectIntruderButton(emoji, emoji == intruder, interaction.user.id))

    await log_action(f"User {interaction.user.id} ({interaction.user.name}) started Eject Intruder")
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

@bot.tree.command(name="scramble", description="Unscramble the word!")
async def cmd_scramble(interaction: discord.Interaction):
    word = random.choice(WORD_SCRAMBLE_WORDS)
    letters = list(word.lower())
    random.shuffle(letters)
    scrambled = ''.join(letters)

    while scrambled == word.lower() and len(word) > 1:
        random.shuffle(letters)
        scrambled = ''.join(letters)

    active_word_scramble[interaction.user.id] = {
        "word": word,
        "scrambled": scrambled.upper()
    }

    embed = discord.Embed(
        title="🔤 Word Scramble",
        description=f"Unscramble this word:\n\n# **{scrambled.upper()}**",
        color=discord.Color.teal()
    )
    embed.add_field(name="Hint", value=f"The word has {len(word)} letters", inline=False)

    view = View(timeout=120)
    guess_btn = Button(label="Submit Answer", style=discord.ButtonStyle.primary)

    async def guess_callback(btn_interaction: discord.Interaction):
        if btn_interaction.user.id != interaction.user.id:
            return await btn_interaction.response.send_message("Start your own game with /scramble!", ephemeral=True)
        await btn_interaction.response.send_modal(WordScrambleModal(interaction.user.id))

    guess_btn.callback = guess_callback
    view.add_item(guess_btn)

    give_up_btn = Button(label="Give Up", style=discord.ButtonStyle.danger)

    async def give_up_callback(btn_interaction: discord.Interaction):
        if btn_interaction.user.id != interaction.user.id:
            return await btn_interaction.response.send_message("This is not your game!", ephemeral=True)
        game = active_word_scramble.get(interaction.user.id)
        if game:
            del active_word_scramble[interaction.user.id]
            embed = discord.Embed(
                title="💀 Game Over",
                description=f"The word was: **{game['word']}**",
                color=discord.Color.red()
            )
            await btn_interaction.response.edit_message(embed=embed, view=None)

    give_up_btn.callback = give_up_callback
    view.add_item(give_up_btn)

    await log_action(f"User {interaction.user.id} ({interaction.user.name}) started Word Scramble")
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

@bot.tree.command(name="cat", description="Get a random cute cat picture!")
async def cmd_cat(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    await log_action(f"User {interaction.user.id} ({interaction.user.name}) requested Cat picture")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.thecatapi.com/v1/images/search") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    cat_url = data[0]["url"]

                    embed = discord.Embed(
                        title="🐱 Here's a kitty!",
                        color=discord.Color.orange()
                    )
                    embed.set_image(url=cat_url)
                    await interaction.followup.send(embed=embed, ephemeral=True)
                else:
                    await interaction.followup.send("Couldn't fetch a cat picture right now. Try again!", ephemeral=True)
    except Exception as e:
        log.exception("Cat API error")
        await interaction.followup.send("Something went wrong fetching a cat picture!", ephemeral=True)

@bot.tree.command(name="funfact", description="Get a random fun fact!")
async def cmd_funfact(interaction: discord.Interaction):
    fact = random.choice(FUN_FACTS_LIST)
    await log_action(f"User {interaction.user.id} ({interaction.user.name}) requested Fun Fact")

    embed = discord.Embed(
        title="🧠 Fun Fact!",
        description=fact,
        color=discord.Color.gold()
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="describe_command", description="Generate a command from description using AI")
@app_commands.describe(
    description="Describe what the command should do",
    api_key="Optional: Your own Gemini or OpenAI API key",
    ai_provider="Optional: Choose 'gemini' or 'openai' (default: gemini)"
)
@app_commands.choices(ai_provider=[
    app_commands.Choice(name="Gemini", value="gemini"),
    app_commands.Choice(name="OpenAI", value="openai"),
])
async def describe_command(
    interaction: discord.Interaction, 
    description: str,
    api_key: str = None,
    ai_provider: app_commands.Choice[str] = None
):
    if not is_admin(interaction):
        return await interaction.response.send_message(f"{EMOJI_LOCK} Admin only.", ephemeral=True)

    guild_id = str(interaction.guild.id)
    await interaction.response.defer(thinking=True, ephemeral=True)

    provider = ai_provider.value if ai_provider else "gemini"

    if api_key:
        cmd_name, code, cmd_desc = await ai_generate_code_with_key(description, api_key, provider)
    else:
        if not gemini_client:
            return await interaction.followup.send(
                "No default AI configured. Please provide your own API key using the `api_key` parameter.",
                ephemeral=True
            )
        cmd_name, code, cmd_desc = await ai_generate_code(description)

    if not cmd_name or not code:
        return await interaction.followup.send(f"Failed to generate command: {cmd_desc}", ephemeral=True)

    success, error = await register_dynamic_command(guild_id, cmd_name, code, cmd_desc)

    if not success:
        return await interaction.followup.send(f"Generated command has errors:\n```{error[:500]}```", ephemeral=True)

    saved = await save_dynamic_command(guild_id, cmd_name, code, cmd_desc)
    if not saved:
        return await interaction.followup.send("Command created but failed to save to storage.", ephemeral=True)

    await sync_all_guild_commands()

    embed = discord.Embed(
        title=f"{EMOJI_THINK} Command Generated!",
        description=f"**Name:** `{cmd_name}`\n**Description:** {cmd_desc}",
        color=discord.Color.green(),
    )
    preview = code if len(code) <= 1000 else code[:1000] + "\n..."
    embed.add_field(name="Code Preview", value=f"```python\n{preview}\n```", inline=False)
    if api_key:
        embed.set_footer(text=f"Generated using your {provider.capitalize()} API key")
    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="create_command", description="Upload a .py file to create a custom command")
@app_commands.describe(file="A .py file that defines async def run(interaction)")
async def create_command(interaction: discord.Interaction, file: discord.Attachment):
    if not is_admin(interaction):
        return await interaction.response.send_message(f"{EMOJI_LOCK} Admin only.", ephemeral=True)

    guild_id = str(interaction.guild.id)

    if not file.filename.endswith('.py'):
        return await interaction.response.send_message("Please upload a .py file!", ephemeral=True)

    await interaction.response.defer(thinking=True, ephemeral=True)

    try:
        code_bytes = await file.read()
        code = code_bytes.decode('utf-8')
        cmd_name = file.filename[:-3].lower().replace(' ', '_').replace('-', '_')

        success, error = await register_dynamic_command(guild_id, cmd_name, code, f"Custom command: {cmd_name}")

        if not success:
            return await interaction.followup.send(f"Command has errors:\n```{error[:500]}```", ephemeral=True)

        saved = await save_dynamic_command(guild_id, cmd_name, code, f"Custom command: {cmd_name}")
        if not saved:
            return await interaction.followup.send("Command created but failed to save to storage.", ephemeral=True)

        await sync_all_guild_commands()
        await interaction.followup.send(f"{EMOJI_THINK} Command `{cmd_name}` created successfully!", ephemeral=True)
    except Exception as e:
        log.exception("create_command error")
        await interaction.followup.send(f"Error creating command: {str(e)}", ephemeral=True)

@bot.tree.command(name="list_commands", description="List all custom commands in this server")
async def list_commands(interaction: discord.Interaction):
    if not is_admin(interaction):
        return await interaction.response.send_message(f"{EMOJI_LOCK} Admin only.", ephemeral=True)

    guild_id = str(interaction.guild.id)

    if guild_id not in dynamic_commands_cache or not dynamic_commands_cache[guild_id]:
        return await interaction.response.send_message("No custom commands in this server.", ephemeral=True)

    commands_list = dynamic_commands_cache[guild_id]
    embed = discord.Embed(
        title=f"{EMOJI_CODE} Custom Commands",
        description=f"Total: {len(commands_list)}",
        color=discord.Color.blue()
    )

    for cmd_name, cmd_data in list(commands_list.items())[:25]:
        embed.add_field(
            name=f"`/{cmd_name}`",
            value=cmd_data.get("description", "No description"),
            inline=False
        )

    if len(commands_list) > 25:
        embed.add_field(name="Note", value=f"Showing 25 of {len(commands_list)} commands", inline=False)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="rename_command", description="Rename a custom command")
@app_commands.describe(old_name="Current command name", new_name="New command name")
async def rename_command(interaction: discord.Interaction, old_name: str, new_name: str):
    if not is_admin(interaction):
        return await interaction.response.send_message(f"{EMOJI_LOCK} Admin only.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)

    guild_id = str(interaction.guild.id)
    old_name = old_name.lower().replace(' ', '_').replace('-', '_')
    new_name = new_name.lower().replace(' ', '_').replace('-', '_')

    if guild_id not in dynamic_commands_cache or old_name not in dynamic_commands_cache[guild_id]:
        return await interaction.followup.send(f"Command `{old_name}` not found.", ephemeral=True)

    cmd_data = dynamic_commands_cache[guild_id][old_name]
    code = cmd_data.get("code", "")
    description = cmd_data.get("description", f"Dynamic command: {new_name}")

    await unregister_dynamic_command(guild_id, old_name)

    success, error = await register_dynamic_command(guild_id, new_name, code, description)
    if not success:
        await register_dynamic_command(guild_id, old_name, code, cmd_data.get("description"))
        await sync_guild_commands(guild_id)
        return await interaction.followup.send(f"Failed to rename command: {error}", ephemeral=True)

    dynamic_commands_cache[guild_id].pop(old_name)
    dynamic_commands_cache[guild_id][new_name] = cmd_data

    config = load_config()
    if "dynamic_commands" not in config:
        config["dynamic_commands"] = {}
    if guild_id not in config["dynamic_commands"]:
        config["dynamic_commands"][guild_id] = {}
    if old_name in config["dynamic_commands"][guild_id]:
        config["dynamic_commands"][guild_id].pop(old_name)
    config["dynamic_commands"][guild_id][new_name] = cmd_data

    if not save_config(config):
        await unregister_dynamic_command(guild_id, new_name)
        await register_dynamic_command(guild_id, old_name, code, cmd_data.get("description"))
        dynamic_commands_cache[guild_id][old_name] = cmd_data
        dynamic_commands_cache[guild_id].pop(new_name)
        await sync_guild_commands(guild_id)
        return await interaction.followup.send("Failed to save rename.", ephemeral=True)

    await sync_guild_commands(guild_id)
    await interaction.followup.send(f"{EMOJI_CHECK} Command renamed: `{old_name}` → `{new_name}`", ephemeral=True)

@bot.tree.command(name="rename_command_description", description="Rename a custom command's description")
@app_commands.describe(command_name="Command name", new_description="New description")
async def rename_command_description(interaction: discord.Interaction, command_name: str, new_description: str):
    if not is_admin(interaction):
        return await interaction.response.send_message(f"{EMOJI_LOCK} Admin only.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)

    guild_id = str(interaction.guild.id)
    command_name = command_name.lower().replace(' ', '_').replace('-', '_')

    if guild_id not in dynamic_commands_cache or command_name not in dynamic_commands_cache[guild_id]:
        return await interaction.followup.send(f"Command `{command_name}` not found.", ephemeral=True)

    cmd_data = dynamic_commands_cache[guild_id][command_name]
    old_description = cmd_data.get("description", "")
    code = cmd_data.get("code", "")

    await unregister_dynamic_command(guild_id, command_name)

    success, error = await register_dynamic_command(guild_id, command_name, code, new_description)
    if not success:
        await register_dynamic_command(guild_id, command_name, code, old_description)
        await sync_guild_commands(guild_id)
        return await interaction.followup.send(f"Failed to update description: {error}", ephemeral=True)

    dynamic_commands_cache[guild_id][command_name]["description"] = new_description

    config = load_config()
    if "dynamic_commands" not in config:
        config["dynamic_commands"] = {}
    if guild_id not in config["dynamic_commands"]:
        config["dynamic_commands"][guild_id] = {}
    if command_name in config["dynamic_commands"][guild_id]:
        config["dynamic_commands"][guild_id][command_name]["description"] = new_description

    if not save_config(config):
        return await interaction.followup.send("Failed to save description.", ephemeral=True)

    await sync_guild_commands(guild_id)
    await interaction.followup.send(f"{EMOJI_CHECK} Description updated for `{command_name}`", ephemeral=True)

@bot.tree.command(name="delete_command", description="Delete a custom command")
@app_commands.describe(command_name="Command name to delete")
async def delete_command(interaction: discord.Interaction, command_name: str):
    if not is_admin(interaction):
        return await interaction.response.send_message(f"{EMOJI_LOCK} Admin only.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)

    guild_id = str(interaction.guild.id)
    command_name = command_name.lower().replace(' ', '_').replace('-', '_')

    if guild_id not in dynamic_commands_cache or command_name not in dynamic_commands_cache[guild_id]:
        return await interaction.followup.send(f"Command `{command_name}` not found.", ephemeral=True)

    await unregister_dynamic_command(guild_id, command_name)

    dynamic_commands_cache[guild_id].pop(command_name)

    config = load_config()
    if "dynamic_commands" in config and guild_id in config["dynamic_commands"]:
        config["dynamic_commands"][guild_id].pop(command_name, None)
        if not save_config(config):
            return await interaction.followup.send("Failed to delete command.", ephemeral=True)

    await sync_guild_commands(guild_id)
    await interaction.followup.send(f"{EMOJI_APPROVED} Command `{command_name}` deleted.", ephemeral=True)

@bot.tree.command(name="userinfo", description="Get information about a user")
@app_commands.describe(user="The user to get information about")
async def cmd_userinfo(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user

    embed = discord.Embed(
        title=f"👤 User Info: {target.display_name}",
        color=target.color if target.color != discord.Color.default() else discord.Color.blue()
    )
    embed.set_thumbnail(url=target.display_avatar.url)
    embed.add_field(name="Username", value=str(target), inline=True)
    embed.add_field(name="ID", value=target.id, inline=True)
    embed.add_field(name="Nickname", value=target.nick or "None", inline=True)
    embed.add_field(name="Account Created", value=f"<t:{int(target.created_at.timestamp())}:R>", inline=True)
    embed.add_field(name="Joined Server", value=f"<t:{int(target.joined_at.timestamp())}:R>" if target.joined_at else "Unknown", inline=True)
    embed.add_field(name="Bot", value="Yes" if target.bot else "No", inline=True)

    roles = [r.mention for r in target.roles if r.name != "@everyone"]
    if roles:
        embed.add_field(name=f"Roles ({len(roles)})", value=", ".join(roles[:10]) + ("..." if len(roles) > 10 else ""), inline=False)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="serverinfo", description="Get information about this server")
@app_commands.describe(server_id="Optional: Server ID to look up (must be discoverable & bot must be in it)")
async def cmd_serverinfo(interaction: discord.Interaction, server_id: str = None):
    if server_id:
        try:
            guild = bot.get_guild(int(server_id))
            if not guild:
                return await interaction.response.send_message("Server not found or bot is not in that server.", ephemeral=True)
        except:
            return await interaction.response.send_message("Invalid server ID.", ephemeral=True)
    else:
        guild = interaction.guild

    embed = discord.Embed(
        title=f"🏠 Server Info: {guild.name}",
        color=discord.Color.blue()
    )
    if guild.icon:
        embed.set_thumbnail(url=guild.icon.url)

    embed.add_field(name="Owner", value=guild.owner.mention if guild.owner else "Unknown", inline=True)
    embed.add_field(name="ID", value=guild.id, inline=True)
    embed.add_field(name="Created", value=f"<t:{int(guild.created_at.timestamp())}:R>", inline=True)
    embed.add_field(name="Members", value=guild.member_count, inline=True)
    embed.add_field(name="Channels", value=len(guild.channels), inline=True)
    embed.add_field(name="Roles", value=len(guild.roles), inline=True)
    embed.add_field(name="Emojis", value=len(guild.emojis), inline=True)
    embed.add_field(name="Boost Level", value=guild.premium_tier, inline=True)
    embed.add_field(name="Boosts", value=guild.premium_subscription_count, inline=True)

    if guild.description:
        embed.add_field(name="Description", value=guild.description, inline=False)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="roleinfo", description="Get information about a role")
@app_commands.describe(role="The role to get information about")
async def cmd_roleinfo(interaction: discord.Interaction, role: discord.Role):
    embed = discord.Embed(
        title=f"🎭 Role Info: {role.name}",
        color=role.color if role.color != discord.Color.default() else discord.Color.blue()
    )
    embed.add_field(name="ID", value=role.id, inline=True)
    embed.add_field(name="Color", value=str(role.color), inline=True)
    embed.add_field(name="Position", value=role.position, inline=True)
    embed.add_field(name="Mentionable", value="Yes" if role.mentionable else "No", inline=True)
    embed.add_field(name="Hoisted", value="Yes" if role.hoist else "No", inline=True)
    embed.add_field(name="Created", value=f"<t:{int(role.created_at.timestamp())}:R>", inline=True)

    perms = [perm.replace("_", " ").title() for perm, value in role.permissions if value]
    if perms:
        embed.add_field(name="Key Permissions", value=", ".join(perms[:15]) + ("..." if len(perms) > 15 else ""), inline=False)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="channelinfo", description="Get information about a channel")
@app_commands.describe(channel="The channel to get information about")
async def cmd_channelinfo(interaction: discord.Interaction, channel: discord.abc.GuildChannel = None):
    target = channel or interaction.channel

    embed = discord.Embed(
        title=f"📺 Channel Info: #{target.name}",
        color=discord.Color.blue()
    )
    embed.add_field(name="ID", value=target.id, inline=True)
    embed.add_field(name="Type", value=str(target.type).replace("_", " ").title(), inline=True)
    embed.add_field(name="Position", value=target.position, inline=True)
    embed.add_field(name="Created", value=f"<t:{int(target.created_at.timestamp())}:R>", inline=True)

    if hasattr(target, "category") and target.category:
        embed.add_field(name="Category", value=target.category.name, inline=True)

    if hasattr(target, "topic") and target.topic:
        embed.add_field(name="Topic", value=target.topic[:100], inline=False)

    if hasattr(target, "slowmode_delay"):
        embed.add_field(name="Slowmode", value=f"{target.slowmode_delay}s" if target.slowmode_delay else "Off", inline=True)

    if hasattr(target, "nsfw"):
        embed.add_field(name="NSFW", value="Yes" if target.nsfw else "No", inline=True)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="emoji-info", description="Get information about an emoji")
@app_commands.describe(emoji="The emoji name or ID")
async def cmd_emojiinfo(interaction: discord.Interaction, emoji: str):
    custom_emoji = None
    for e in interaction.guild.emojis:
        if str(e.id) in emoji or e.name.lower() == emoji.lower():
            custom_emoji = e
            break

    if not custom_emoji:
        return await interaction.response.send_message("Emoji not found. Make sure it's a custom emoji from this server.", ephemeral=True)

    embed = discord.Embed(
        title=f"😀 Emoji Info: {custom_emoji.name}",
        color=discord.Color.blue()
    )
    embed.set_thumbnail(url=custom_emoji.url)
    embed.add_field(name="ID", value=custom_emoji.id, inline=True)
    embed.add_field(name="Name", value=custom_emoji.name, inline=True)
    embed.add_field(name="Animated", value="Yes" if custom_emoji.animated else "No", inline=True)
    embed.add_field(name="Created", value=f"<t:{int(custom_emoji.created_at.timestamp())}:R>", inline=True)
    embed.add_field(name="URL", value=f"[Link]({custom_emoji.url})", inline=True)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="purge", description="Delete messages from this channel")
@app_commands.describe(amount="Number of messages to delete (1-100)")
@app_commands.checks.has_permissions(manage_messages=True)
async def cmd_purge(interaction: discord.Interaction, amount: int):
    if amount < 1 or amount > 100:
        return await interaction.response.send_message("Amount must be between 1 and 100.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)

    try:
        deleted = await interaction.channel.purge(limit=amount)
        await interaction.followup.send(f"🗑️ Deleted {len(deleted)} messages.", ephemeral=True)
    except discord.Forbidden:
        await interaction.followup.send("I don't have permission to delete messages in this channel.", ephemeral=True)
    except discord.HTTPException as e:
        await interaction.followup.send(f"Failed to delete messages: {e}", ephemeral=True)

@bot.tree.command(name="purge-apps", description="Delete bot messages from this channel")
@app_commands.describe(amount="Number of bot messages to delete (1-100)")
@app_commands.checks.has_permissions(manage_messages=True)
async def cmd_purge_apps(interaction: discord.Interaction, amount: int):
    if amount < 1 or amount > 100:
        return await interaction.response.send_message("Amount must be between 1 and 100.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)

    def is_bot(message):
        return message.author.bot

    try:
        deleted = await interaction.channel.purge(limit=amount * 2, check=is_bot)
        await interaction.followup.send(f"🗑️ Deleted {len(deleted)} bot messages.", ephemeral=True)
    except discord.Forbidden:
        await interaction.followup.send("I don't have permission to delete messages in this channel.", ephemeral=True)
    except discord.HTTPException as e:
        await interaction.followup.send(f"Failed to delete messages: {e}", ephemeral=True)

@bot.tree.command(name="purge-humans", description="Delete human messages from this channel")
@app_commands.describe(amount="Number of human messages to delete (1-100)")
@app_commands.checks.has_permissions(manage_messages=True)
async def cmd_purge_humans(interaction: discord.Interaction, amount: int):
    if amount < 1 or amount > 100:
        return await interaction.response.send_message("Amount must be between 1 and 100.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)

    def is_human(message):
        return not message.author.bot

    try:
        deleted = await interaction.channel.purge(limit=amount * 2, check=is_human)
        await interaction.followup.send(f"🗑️ Deleted {len(deleted)} human messages.", ephemeral=True)
    except discord.Forbidden:
        await interaction.followup.send("I don't have permission to delete messages in this channel.", ephemeral=True)
    except discord.HTTPException as e:
        await interaction.followup.send(f"Failed to delete messages: {e}", ephemeral=True)

@bot.tree.command(name="kick", description="Kick a member from the server")
@app_commands.describe(user="The user to kick", reason="Reason for kicking")
@app_commands.checks.has_permissions(kick_members=True)
async def cmd_kick(interaction: discord.Interaction, user: discord.Member, reason: str = None):
    if user.id == interaction.user.id:
        return await interaction.response.send_message("You can't kick yourself.", ephemeral=True)

    if user.top_role >= interaction.user.top_role and interaction.user.id != interaction.guild.owner_id:
        return await interaction.response.send_message("You can't kick someone with a higher or equal role.", ephemeral=True)

    if not interaction.guild.me.guild_permissions.kick_members:
        return await interaction.response.send_message("I don't have permission to kick members.", ephemeral=True)

    if user.top_role >= interaction.guild.me.top_role:
        return await interaction.response.send_message("I can't kick someone with a higher or equal role than me.", ephemeral=True)

    try:
        await user.kick(reason=reason or f"Kicked by {interaction.user}")
    except discord.Forbidden:
        return await interaction.response.send_message("I don't have permission to kick this member.", ephemeral=True)
    except discord.HTTPException as e:
        return await interaction.response.send_message(f"Failed to kick member: {e}", ephemeral=True)

    embed = discord.Embed(
        title="👢 Member Kicked",
        description=f"{user.mention} has been kicked.",
        color=discord.Color.orange()
    )
    if reason:
        embed.add_field(name="Reason", value=reason)
    embed.add_field(name="Kicked by", value=interaction.user.mention)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="ban", description="Ban a member from the server")
@app_commands.describe(user="The user to ban", duration="Ban duration (e.g., 7d, 1h, 30m) - leave empty for permanent", reason="Reason for banning")
@app_commands.checks.has_permissions(ban_members=True)
async def cmd_ban(interaction: discord.Interaction, user: discord.Member, duration: str = None, reason: str = None):
    if user.id == interaction.user.id:
        return await interaction.response.send_message("You can't ban yourself.", ephemeral=True)

    if user.top_role >= interaction.user.top_role and interaction.user.id != interaction.guild.owner_id:
        return await interaction.response.send_message("You can't ban someone with a higher or equal role.", ephemeral=True)

    if not interaction.guild.me.guild_permissions.ban_members:
        return await interaction.response.send_message("I don't have permission to ban members.", ephemeral=True)

    if user.top_role >= interaction.guild.me.top_role:
        return await interaction.response.send_message("I can't ban someone with a higher or equal role than me.", ephemeral=True)

    try:
        await user.ban(reason=reason or f"Banned by {interaction.user}")
    except discord.Forbidden:
        return await interaction.response.send_message("I don't have permission to ban this member.", ephemeral=True)
    except discord.HTTPException as e:
        return await interaction.response.send_message(f"Failed to ban member: {e}", ephemeral=True)

    embed = discord.Embed(
        title="🔨 Member Banned",
        description=f"{user.mention} has been banned.",
        color=discord.Color.red()
    )
    if duration:
        embed.add_field(name="Duration", value=duration)
    if reason:
        embed.add_field(name="Reason", value=reason)
    embed.add_field(name="Banned by", value=interaction.user.mention)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="mute", description="Mute a member (timeout)")
@app_commands.describe(user="The user to mute", duration="Mute duration (e.g., 7d, 1h, 30m)", reason="Reason for muting")
@app_commands.checks.has_permissions(moderate_members=True)
async def cmd_mute(interaction: discord.Interaction, user: discord.Member, duration: str, reason: str = None):
    if user.id == interaction.user.id:
        return await interaction.response.send_message("You can't mute yourself.", ephemeral=True)

    if user.top_role >= interaction.user.top_role and interaction.user.id != interaction.guild.owner_id:
        return await interaction.response.send_message("You can't mute someone with a higher or equal role.", ephemeral=True)

    if not interaction.guild.me.guild_permissions.moderate_members:
        return await interaction.response.send_message("I don't have permission to timeout members.", ephemeral=True)

    if user.top_role >= interaction.guild.me.top_role:
        return await interaction.response.send_message("I can't mute someone with a higher or equal role than me.", ephemeral=True)

    duration_delta = parse_duration(duration)
    if not duration_delta:
        return await interaction.response.send_message("Invalid duration format. Use formats like: 1h, 30m, 7d", ephemeral=True)

    if duration_delta > timedelta(days=28):
        return await interaction.response.send_message("Maximum mute duration is 28 days.", ephemeral=True)

    try:
        await user.timeout(duration_delta, reason=reason or f"Muted by {interaction.user}")
    except discord.Forbidden:
        return await interaction.response.send_message("I don't have permission to timeout this member.", ephemeral=True)
    except discord.HTTPException as e:
        return await interaction.response.send_message(f"Failed to mute member: {e}", ephemeral=True)

    embed = discord.Embed(
        title="🔇 Member Muted",
        description=f"{user.mention} has been muted.",
        color=discord.Color.orange()
    )
    embed.add_field(name="Duration", value=duration)
    if reason:
        embed.add_field(name="Reason", value=reason)
    embed.add_field(name="Muted by", value=interaction.user.mention)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="timeout", description="Timeout a member")
@app_commands.describe(user="The user to timeout", duration="Timeout duration (e.g., 7d, 1h, 30m)", reason="Reason for timeout")
@app_commands.checks.has_permissions(moderate_members=True)
async def cmd_timeout(interaction: discord.Interaction, user: discord.Member, duration: str, reason: str = None):
    if user.id == interaction.user.id:
        return await interaction.response.send_message("You can't timeout yourself.", ephemeral=True)

    if user.top_role >= interaction.user.top_role and interaction.user.id != interaction.guild.owner_id:
        return await interaction.response.send_message("You can't timeout someone with a higher or equal role.", ephemeral=True)

    if not interaction.guild.me.guild_permissions.moderate_members:
        return await interaction.response.send_message("I don't have permission to timeout members.", ephemeral=True)

    if user.top_role >= interaction.guild.me.top_role:
        return await interaction.response.send_message("I can't timeout someone with a higher or equal role than me.", ephemeral=True)

    duration_delta = parse_duration(duration)
    if not duration_delta:
        return await interaction.response.send_message("Invalid duration format. Use formats like: 1h, 30m, 7d", ephemeral=True)

    if duration_delta > timedelta(days=28):
        return await interaction.response.send_message("Maximum timeout duration is 28 days.", ephemeral=True)

    try:
        await user.timeout(duration_delta, reason=reason or f"Timed out by {interaction.user}")
    except discord.Forbidden:
        return await interaction.response.send_message("I don't have permission to timeout this member.", ephemeral=True)
    except discord.HTTPException as e:
        return await interaction.response.send_message(f"Failed to timeout member: {e}", ephemeral=True)

    embed = discord.Embed(
        title="⏰ Member Timed Out",
        description=f"{user.mention} has been timed out.",
        color=discord.Color.orange()
    )
    embed.add_field(name="Duration", value=duration)
    if reason:
        embed.add_field(name="Reason", value=reason)
    embed.add_field(name="Timed out by", value=interaction.user.mention)

    await interaction.response.send_message(embed=embed, ephemeral=True)

def parse_duration(duration_str: str) -> Optional[timedelta]:
    match = re.match(r'^(\d+)([smhd])$', duration_str.lower())
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2)

    if unit == 's':
        return timedelta(seconds=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    return None

@bot.tree.command(name="leave", description="Make the bot leave this server (Server owner only)")
async def cmd_leave(interaction: discord.Interaction):
    if not is_owner(interaction):
        return await interaction.response.send_message("Only the server owner can use this command.", ephemeral=True)

    await interaction.response.send_message("👋 Goodbye! Leaving the server...", ephemeral=True)
    await interaction.guild.leave()

@bot.tree.command(name="lock", description="Lock a channel so only users with manage_channels can send messages")
@app_commands.describe(duration="Lock duration (number)", duration_type="Duration type")
@app_commands.choices(duration_type=[
    app_commands.Choice(name="Minutes", value="minutes"),
    app_commands.Choice(name="Hours", value="hours"),
    app_commands.Choice(name="Days", value="days")
])
@app_commands.checks.has_permissions(manage_channels=True)
async def cmd_lock(interaction: discord.Interaction, duration: int = None, duration_type: str = None):
    channel = interaction.channel

    overwrite = channel.overwrites_for(interaction.guild.default_role)
    overwrite.send_messages = False
    await channel.set_permissions(interaction.guild.default_role, overwrite=overwrite)

    if duration and duration_type:
        if duration_type == "minutes":
            unlock_time = datetime.now() + timedelta(minutes=duration)
        elif duration_type == "hours":
            unlock_time = datetime.now() + timedelta(hours=duration)
        else:
            unlock_time = datetime.now() + timedelta(days=duration)

        locked_channels[channel.id] = {
            "unlock_time": unlock_time,
            "guild_id": interaction.guild.id
        }

        embed = discord.Embed(
            title="🔒 Channel Locked",
            description=f"This channel has been locked for {duration} {duration_type}.",
            color=discord.Color.red()
        )
        embed.add_field(name="Unlocks", value=f"<t:{int(unlock_time.timestamp())}:R>")
    else:
        embed = discord.Embed(
            title="🔒 Channel Locked",
            description="This channel has been locked indefinitely.",
            color=discord.Color.red()
        )

    embed.add_field(name="Locked by", value=interaction.user.mention)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="unlock", description="Unlock a locked channel")
@app_commands.checks.has_permissions(manage_channels=True)
async def cmd_unlock(interaction: discord.Interaction):
    channel = interaction.channel

    overwrite = channel.overwrites_for(interaction.guild.default_role)
    overwrite.send_messages = None
    await channel.set_permissions(interaction.guild.default_role, overwrite=overwrite)

    if channel.id in locked_channels:
        del locked_channels[channel.id]

    embed = discord.Embed(
        title="🔓 Channel Unlocked",
        description="This channel has been unlocked.",
        color=discord.Color.green()
    )
    embed.add_field(name="Unlocked by", value=interaction.user.mention)
    await interaction.response.send_message(embed=embed, ephemeral=True)

class UndoButton(Button):
    def __init__(self, undo_action: str, undo_data: dict):
        super().__init__(label="Undo", style=discord.ButtonStyle.danger, emoji="↩️")
        self.undo_action = undo_action
        self.undo_data = undo_data

    async def callback(self, interaction: discord.Interaction):
        if not interaction.user.guild_permissions.manage_guild:
            return await interaction.response.send_message("You need Manage Server permission to undo.", ephemeral=True)

        try:
            if self.undo_action == "log_enable":
                guild_id = str(interaction.guild.id)
                log_type = self.undo_data.get("log_type")
                if guild_id in guild_log_settings and log_type in guild_log_settings[guild_id]:
                    del guild_log_settings[guild_id][log_type]
                    await interaction.response.send_message(f"✅ Undone: {log_type} logging disabled.", ephemeral=True)
                else:
                    await interaction.response.send_message("Nothing to undo.", ephemeral=True)
            else:
                await interaction.response.send_message("This action cannot be undone.", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f"Failed to undo: {e}", ephemeral=True)

@bot.tree.command(name="log", description="Set up logging for server events")
@app_commands.describe(log_type="Type of events to log", channel="Channel to send logs to")
@app_commands.choices(log_type=[
    app_commands.Choice(name="Emoji", value="emoji"),
    app_commands.Choice(name="Channel", value="channel"),
    app_commands.Choice(name="Server", value="server"),
    app_commands.Choice(name="Role", value="role"),
    app_commands.Choice(name="Messages", value="messages"),
    app_commands.Choice(name="Mod Actions", value="mod-actions")
])
@app_commands.checks.has_permissions(manage_guild=True)
async def cmd_log(interaction: discord.Interaction, log_type: str, channel: discord.TextChannel):
    guild_id = str(interaction.guild.id)

    if guild_id not in guild_log_settings:
        guild_log_settings[guild_id] = {}

    guild_log_settings[guild_id][log_type] = channel.id

    embed = discord.Embed(
        title="📋 Logging Enabled",
        description=f"**{log_type.title()}** events will now be logged to {channel.mention}",
        color=discord.Color.green()
    )

    view = View(timeout=300)
    undo_btn = UndoButton("log_enable", {"log_type": log_type})
    view.add_item(undo_btn)

    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

@bot.tree.command(name="unlog", description="Disable logging for server events")
@app_commands.describe(log_type="Type of events to stop logging")
@app_commands.choices(log_type=[
    app_commands.Choice(name="Emoji", value="emoji"),
    app_commands.Choice(name="Channel", value="channel"),
    app_commands.Choice(name="Server", value="server"),
    app_commands.Choice(name="Role", value="role"),
    app_commands.Choice(name="Messages", value="messages"),
    app_commands.Choice(name="Mod Actions", value="mod-actions"),
    app_commands.Choice(name="All", value="all")
])
@app_commands.checks.has_permissions(manage_guild=True)
async def cmd_unlog(interaction: discord.Interaction, log_type: str):
    guild_id = str(interaction.guild.id)

    if guild_id not in guild_log_settings:
        return await interaction.response.send_message("No logging is set up for this server.", ephemeral=True)

    if log_type == "all":
        guild_log_settings[guild_id] = {}
        msg = "All logging has been disabled."
    else:
        if log_type in guild_log_settings[guild_id]:
            del guild_log_settings[guild_id][log_type]
            msg = f"**{log_type.title()}** logging has been disabled."
        else:
            return await interaction.response.send_message(f"{log_type.title()} logging was not enabled.", ephemeral=True)

    embed = discord.Embed(
        title="📋 Logging Disabled",
        description=msg,
        color=discord.Color.orange()
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)

class SendMessageModal(Modal, title="Send Anonymous Message"):
    message_content = TextInput(
        label="Message",
        style=discord.TextStyle.paragraph,
        placeholder="Enter your message here...",
        required=True,
        max_length=2000
    )

    embed_color = TextInput(
        label="Embed Hex Color (optional)",
        placeholder="e.g., #FF5733 or leave empty for no embed",
        required=False,
        max_length=7
    )

    sticky_note = TextInput(
        label="Sticky Note? (Y/N)",
        placeholder="Y for sticky, N or empty for normal",
        required=False,
        max_length=1
    )

    def __init__(self, target_user: discord.Member = None):
        super().__init__()
        self.target_user = target_user

    async def on_submit(self, interaction: discord.Interaction):
        # Parse sticky note option
        is_sticky = self.sticky_note.value.upper() == "Y" if self.sticky_note.value else False

        # Check if this is a DM
        if self.target_user:
            # Send DM
            try:
                if self.embed_color.value:
                    try:
                        color_value = self.embed_color.value.strip()
                        if not color_value.startswith("#"):
                            color_value = f"#{color_value}"
                        embed = discord.Embed(
                            description=self.message_content.value,
                            color=discord.Color.from_str(color_value)
                        )
                        embed.set_footer(text=f"Sent by {interaction.user.name} via bot")
                        await self.target_user.send(embed=embed)
                    except:
                        msg = f"{self.message_content.value}\n\n*Sent by {interaction.user.name} via bot*"
                        await self.target_user.send(msg)
                else:
                    msg = f"{self.message_content.value}\n\n*Sent by {interaction.user.name} via bot*"
                    await self.target_user.send(msg)
                
                await interaction.response.send_message(f"DM sent to {self.target_user.mention}!", ephemeral=True)
            except discord.Forbidden:
                await interaction.response.send_message(f"Cannot send DM to {self.target_user.mention}. They may have DMs disabled.", ephemeral=True)
            except Exception as e:
                await interaction.response.send_message(f"Error sending DM: {e}", ephemeral=True)
        else:
            # Send to channel
            channel = interaction.channel

            # Create message content
            if self.embed_color.value:
                # Try to parse hex color
                try:
                    color_value = self.embed_color.value.strip()
                    if not color_value.startswith("#"):
                        color_value = f"#{color_value}"
                    embed = discord.Embed(
                        description=self.message_content.value,
                        color=discord.Color.from_str(color_value)
                    )
                    sent_msg = await channel.send(embed=embed)
                except:
                    # Invalid color, send as normal message
                    sent_msg = await channel.send(self.message_content.value)
            else:
                sent_msg = await channel.send(self.message_content.value)

            # Handle sticky note
            if is_sticky:
                # Store sticky note info
                guild_id = str(interaction.guild.id)
                config = load_config()
                if "sticky_notes" not in config:
                    config["sticky_notes"] = {}
                if guild_id not in config["sticky_notes"]:
                    config["sticky_notes"][guild_id] = {}

                config["sticky_notes"][guild_id][str(channel.id)] = {
                    "message_id": sent_msg.id,
                    "content": self.message_content.value,
                    "embed_color": self.embed_color.value if self.embed_color.value else None
                }
                save_config(config)

            await interaction.response.send_message("Message sent!" + (" (Sticky note enabled)" if is_sticky else ""), ephemeral=True)

@bot.tree.command(name="send", description="Send an anonymous message in this channel or DM a user")
@app_commands.describe(user="Optional: User to send DM to (will show who sent it)")
@app_commands.checks.has_permissions(manage_messages=True)
async def cmd_send(interaction: discord.Interaction, user: discord.Member = None):
    await interaction.response.send_modal(SendMessageModal(target_user=user))



@bot.event
async def on_member_join(member):
    guild_id = str(member.guild.id)
    config = load_config()

    if "welcome_channels" in config and guild_id in config["welcome_channels"]:
        channel_id = config["welcome_channels"][guild_id]
        channel = member.guild.get_channel(channel_id)
        if channel:
            try:
                image_buffer = create_welcome_image(member, is_join=True)
                file = discord.File(image_buffer, filename="welcome.png")
                await channel.send(file=file)
            except Exception as e:
                log.error(f"Failed to send welcome message: {e}")

@bot.event
async def on_member_leave(member):
    guild_id = str(member.guild.id)
    config = load_config()

    if "leave_channels" in config and guild_id in config["leave_channels"]:
        channel_id = config["leave_channels"][guild_id]
        channel = member.guild.get_channel(channel_id)
        if channel:
            try:
                image_buffer = create_welcome_image(member, is_join=False)
                file = discord.File(image_buffer, filename="goodbye.png")
                await channel.send(file=file)
            except Exception as e:
                log.error(f"Failed to send goodbye message: {e}")

@bot.event
async def on_guild_channel_create(channel):
    guild_id = str(channel.guild.id)
    if guild_id in guild_log_settings and "channel" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["channel"])
        if log_channel:
            embed = discord.Embed(
                title="📺 Channel Created",
                description=f"**{channel.name}** was created",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Type", value=str(channel.type))
            embed.add_field(name="ID", value=channel.id)
            await log_channel.send(embed=embed)

@bot.event
async def on_guild_channel_delete(channel):
    guild_id = str(channel.guild.id)
    if guild_id in guild_log_settings and "channel" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["channel"])
        if log_channel:
            embed = discord.Embed(
                title="📺 Channel Deleted",
                description=f"**{channel.name}** was deleted",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Type", value=str(channel.type))
            embed.add_field(name="ID", value=channel.id)
            await log_channel.send(embed=embed)

@bot.event
async def on_guild_channel_update(before, after):
    guild_id = str(after.guild.id)
    if guild_id in guild_log_settings and "channel" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["channel"])
        if log_channel:
            changes = []
            if before.name != after.name:
                changes.append(f"Name: `{before.name}` → `{after.name}`")
            if hasattr(before, "topic") and hasattr(after, "topic") and before.topic != after.topic:
                changes.append(f"Topic changed")

            if changes:
                embed = discord.Embed(
                    title="📺 Channel Updated",
                    description=f"**{after.name}** was updated",
                    color=discord.Color.orange(),
                    timestamp=datetime.now()
                )
                embed.add_field(name="Changes", value="\n".join(changes))
                await log_channel.send(embed=embed)

@bot.event
async def on_guild_role_create(role):
    guild_id = str(role.guild.id)
    if guild_id in guild_log_settings and "role" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["role"])
        if log_channel:
            embed = discord.Embed(
                title="🎭 Role Created",
                description=f"**{role.name}** was created",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Color", value=str(role.color))
            embed.add_field(name="ID", value=role.id)
            await log_channel.send(embed=embed)

@bot.event
async def on_guild_role_delete(role):
    guild_id = str(role.guild.id)
    if guild_id in guild_log_settings and "role" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["role"])
        if log_channel:
            embed = discord.Embed(
                title="🎭 Role Deleted",
                description=f"**{role.name}** was deleted",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            embed.add_field(name="ID", value=role.id)
            await log_channel.send(embed=embed)

@bot.event
async def on_guild_role_update(before, after):
    guild_id = str(after.guild.id)
    if guild_id in guild_log_settings and "role" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["role"])
        if log_channel:
            changes = []
            if before.name != after.name:
                changes.append(f"Name: `{before.name}` → `{after.name}`")
            if before.color != after.color:
                changes.append(f"Color: `{before.color}` → `{after.color}`")

            if changes:
                embed = discord.Embed(
                    title="🎭 Role Updated",
                    description=f"**{after.name}** was updated",
                    color=discord.Color.orange(),
                    timestamp=datetime.now()
                )
                embed.add_field(name="Changes", value="\n".join(changes))
                await log_channel.send(embed=embed)

@bot.event
async def on_guild_emojis_update(guild, before, after):
    guild_id = str(guild.id)
    if guild_id in guild_log_settings and "emoji" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["emoji"])
        if log_channel:
            before_set = set(before)
            after_set = set(after)

            added = after_set - before_set
            removed = before_set - after_set

            for emoji in added:
                embed = discord.Embed(
                    title="😀 Emoji Added",
                    description=f"**{emoji.name}** was added",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                embed.set_thumbnail(url=emoji.url)
                await log_channel.send(embed=embed)

            for emoji in removed:
                embed = discord.Embed(
                    title="😀 Emoji Removed",
                    description=f"**{emoji.name}** was removed",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )
                await log_channel.send(embed=embed)

class UndoServerChangeButton(Button):
    def __init__(self, guild_id: int, change_type: str, old_value: str):
        super().__init__(label="Undo", style=discord.ButtonStyle.danger, emoji="↩️")
        self.guild_id = guild_id
        self.change_type = change_type
        self.old_value = old_value

    async def callback(self, interaction: discord.Interaction):
        if not interaction.user.guild_permissions.manage_guild:
            return await interaction.response.send_message("You need Manage Server permission to undo.", ephemeral=True)

        try:
            guild = bot.get_guild(self.guild_id)
            if not guild:
                return await interaction.response.send_message("Guild not found.", ephemeral=True)

            if self.change_type == "name":
                await guild.edit(name=self.old_value)
                await interaction.response.send_message(f"✅ Server name reverted to: {self.old_value}", ephemeral=True)
            elif self.change_type == "description":
                await guild.edit(description=self.old_value)
                await interaction.response.send_message(f"✅ Server description reverted.", ephemeral=True)

            # Disable button
            for item in self.view.children:
                item.disabled = True
            await interaction.message.edit(view=self.view)
        except Exception as e:
            await interaction.response.send_message(f"Failed to undo: {e}", ephemeral=True)

@bot.event
async def on_guild_update(before, after):
    guild_id = str(after.id)
    if guild_id in guild_log_settings and "server" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["server"])
        if log_channel:
            changes = []
            view = View(timeout=None)
            
            if before.name != after.name:
                changes.append(f"Name: `{before.name}` → `{after.name}`")
                view.add_item(UndoServerChangeButton(after.id, "name", before.name))
            if before.icon != after.icon:
                changes.append("Server icon changed")
            if before.description != after.description:
                changes.append("Description changed")
                view.add_item(UndoServerChangeButton(after.id, "description", before.description or ""))

            if changes:
                embed = discord.Embed(
                    title="🏠 Server Updated",
                    description=f"**{after.name}** was updated",
                    color=discord.Color.orange(),
                    timestamp=datetime.now()
                )
                embed.add_field(name="Changes", value="\n".join(changes))
                
                if len(view.children) > 0:
                    await log_channel.send(embed=embed, view=view)
                else:
                    await log_channel.send(embed=embed)

@bot.event
async def on_message_delete(message):
    if message.author.bot:
        return
    guild_id = str(message.guild.id) if message.guild else None
    if guild_id and guild_id in guild_log_settings and "messages" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["messages"])
        if log_channel:
            embed = discord.Embed(
                title="🗑️ Message Deleted",
                description=f"Message by {message.author.mention} in {message.channel.mention}",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Note", value="Message content not available (requires Message Content Intent)")
            await log_channel.send(embed=embed)

@bot.event
async def on_message_edit(before, after):
    if before.author.bot:
        return
    guild_id = str(before.guild.id) if before.guild else None
    if guild_id and guild_id in guild_log_settings and "messages" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["messages"])
        if log_channel:
            embed = discord.Embed(
                title="✏️ Message Edited",
                description=f"Message by {before.author.mention} in {before.channel.mention}",
                color=discord.Color.orange(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Note", value="Message content not available (requires Message Content Intent)")
            await log_channel.send(embed=embed)

@bot.event
async def on_member_ban(guild, user):
    guild_id = str(guild.id)
    if guild_id in guild_log_settings and "mod-actions" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["mod-actions"])
        if log_channel:
            embed = discord.Embed(
                title="🔨 Member Banned",
                description=f"**{user}** was banned from the server",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            embed.add_field(name="User ID", value=user.id)
            await log_channel.send(embed=embed)

@bot.event
async def on_member_unban(guild, user):
    guild_id = str(guild.id)
    if guild_id in guild_log_settings and "mod-actions" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["mod-actions"])
        if log_channel:
            embed = discord.Embed(
                title="🔓 Member Unbanned",
                description=f"**{user}** was unbanned from the server",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            embed.add_field(name="User ID", value=user.id)
            await log_channel.send(embed=embed)

@bot.event
async def on_member_remove(member):
    guild_id = str(member.guild.id)
    if guild_id in guild_log_settings and "mod-actions" in guild_log_settings[guild_id]:
        log_channel = bot.get_channel(guild_log_settings[guild_id]["mod-actions"])
        if log_channel:
            embed = discord.Embed(
                title="👋 Member Left/Kicked",
                description=f"**{member}** left the server",
                color=discord.Color.orange(),
                timestamp=datetime.now()
            )
            embed.add_field(name="User ID", value=member.id)
            await log_channel.send(embed=embed)

@tasks.loop(minutes=1)
async def check_channel_unlocks():
    now = datetime.now()
    to_unlock = []

    for channel_id, data in locked_channels.items():
        if now >= data["unlock_time"]:
            to_unlock.append(channel_id)

    for channel_id in to_unlock:
        try:
            channel = bot.get_channel(channel_id)
            if channel:
                overwrite = channel.overwrites_for(channel.guild.default_role)
                overwrite.send_messages = None
                await channel.set_permissions(channel.guild.default_role, overwrite=overwrite)

                embed = discord.Embed(
                    title="🔓 Channel Auto-Unlocked",
                    description="This channel has been automatically unlocked.",
                    color=discord.Color.green()
                )
                await channel.send(embed=embed)

            del locked_channels[channel_id]
        except Exception as e:
            log.error(f"Failed to unlock channel {channel_id}: {e}")

@tasks.loop(minutes=5)
async def xp_cache_flush():
    """Periodically flush XP cache to disk"""
    if flush_xp_cache():
        log.debug("XP cache flushed successfully")
    else:
        log.warning("Failed to flush XP cache")

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        missing = ", ".join(error.missing_permissions)
        await interaction.response.send_message(
            f"🔒 You need the following permissions to use this command: **{missing}**",
            ephemeral=True
        )
    elif isinstance(error, app_commands.BotMissingPermissions):
        missing = ", ".join(error.missing_permissions)
        await interaction.response.send_message(
            f"⚠️ I need the following permissions: **{missing}**",
            ephemeral=True
        )
    else:
        log.exception(f"Command error: {error}")
        try:
            await interaction.response.send_message(
                "An error occurred while executing this command.",
                ephemeral=True
            )
        except:
            pass

@bot.event
async def on_message(message):
    if message.author.bot or not message.guild:
        return

    guild_id = str(message.guild.id)

    # Handle sticky notes
    config = load_config()
    if "sticky_notes" in config and guild_id in config["sticky_notes"]:
        channel_id = str(message.channel.id)
        if channel_id in config["sticky_notes"][guild_id]:
            sticky_data = config["sticky_notes"][guild_id][channel_id]
            try:
                # Delete old sticky message
                old_msg = await message.channel.fetch_message(sticky_data["message_id"])
                await old_msg.delete()

                # Resend at bottom
                if sticky_data.get("embed_color"):
                    embed = discord.Embed(
                        description=sticky_data["content"],
                        color=discord.Color.from_str(sticky_data["embed_color"])
                    )
                    new_msg = await message.channel.send(embed=embed)
                else:
                    new_msg = await message.channel.send(sticky_data["content"])

                # Update message ID
                config["sticky_notes"][guild_id][channel_id]["message_id"] = new_msg.id
                save_config(config)
            except:
                pass

    level_config = get_guild_level_config(guild_id)

    if message.channel.id in level_config.get("ignored_channels", []):
        await bot.process_commands(message)
        return

    user_roles = [r.id for r in message.author.roles]
    if any(r in level_config.get("ignored_roles", []) for r in user_roles):
        await bot.process_commands(message)
        return

    current_xp = get_user_xp(guild_id, message.author.id)
    old_level = current_xp // level_config.get("level_xp", 100) if level_config.get("level_xp", 100) > 0 else 0

    new_xp = current_xp + level_config.get("xp_per_message", 10)
    save_user_xp(guild_id, message.author.id, new_xp)

    new_level = new_xp // level_config.get("level_xp", 100) if level_config.get("level_xp", 100) > 0 else 0

    if new_level > old_level:
        level_roles = level_config.get("level_roles", {})
        if str(new_level) in level_roles:
            role_id = level_roles[str(new_level)]
            try:
                role = message.guild.get_role(role_id)
                if role:
                    await message.author.add_roles(role)
                    embed = discord.Embed(
                        title=f"🎉 Level Up!",
                        description=f"**{message.author.mention}** reached level **{new_level}**!",
                        color=discord.Color.gold()
                    )
                    embed.add_field(name="Reward", value=f"Gained role: {role.mention}")
                    await message.channel.send(embed=embed)
            except Exception as e:
                log.error(f"Failed to assign role: {e}")
        else:
            try:
                embed = discord.Embed(
                    title=f"🎉 Level Up!",
                    description=f"**{message.author.mention}** reached level **{new_level}**!",
                    color=discord.Color.gold()
                )
                await message.channel.send(embed=embed)
            except:
                pass

    await bot.process_commands(message)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)