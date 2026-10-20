from utils.instruction_generator import InstructionGenerator


def test_generate_instruction_handles_empty_object_and_container_lists():
    instruction = InstructionGenerator.generate_instruction(
        {"task": "place", "objects": [], "containers": []}
    )

    assert isinstance(instruction, str)
    assert instruction.endswith(".")
