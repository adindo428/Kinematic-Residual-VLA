# -*- coding: utf-8 -*-
#
# @File:   instruction_generator.py
# @Author: Haozhe Xie
# @Date:   2025-05-31 19:42:51
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2026-01-11 20:40:43
# @Email:  root@haozhexie.com

import json
import random


class InstructionGenerator:
    @staticmethod
    def generate_instruction(inst_metadata):
        if isinstance(inst_metadata, str):
            inst_metadata = json.loads(inst_metadata)

        instruction = inst_metadata.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()

        tmpl = InstructionGenerator._get_instruction_template(inst_metadata["task"])
        object_desc = InstructionGenerator._choice_or_default(inst_metadata, "objects")
        container_desc = InstructionGenerator._choice_or_default(
            inst_metadata, "containers"
        )
        return tmpl.format_map({"object": object_desc, "container": container_desc})

    @staticmethod
    def _choice_or_default(inst_metadata, key, default=""):
        choices = inst_metadata.get(key)
        if isinstance(choices, str):
            return choices
        if not choices:
            return default
        return random.choice(choices)

    @staticmethod
    def _get_instruction_template(task):
        task = task.replace("_", "-")
        pick_action = random.choice(
            ["pick up", "grasp", "catch", "grab", "get hold of"]
        )
        place_action = random.choice(
            ["place on", "put on", "set on", "position on", "return to", "deposit in"]
        )
        if task == "pick":
            return f"{pick_action} the {{object}}."
        elif task in ["place", "long-horizon"]:
            return f"{pick_action} the {{object}} and {place_action} the {{container}}."
        else:
            raise ValueError(f"Unknown task: {task}")
