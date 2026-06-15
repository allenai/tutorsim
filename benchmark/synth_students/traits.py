from typing import Optional, Dict, Any, List
# Original synth-students imports below; rewired to our local equivalents to
# avoid pulling synth-students.src wholesale. The ModelWrapper interface
# TraitGenerator needs is `.call(non_system_messages, system_prompt, **kwargs)`;
# we satisfy it via benchmark.synth_students._adapter.ModelWrapperAdapter.
# Source: ../synth-students/src/students/traits.py (verbatim except imports).
from benchmark.synth_students._adapter import ModelWrapperAdapter as ModelWrapper
from benchmark.synth_students._adapter import get_history_str
from benchmark.synth_students.dimension import ALL_DIMENSION_NAMES, ALL_DIMENSIONS, get_dimension_description


def get_default_trait_modes(
    num_sentences: Optional[List] = [
        1,
        2,
        3,
        4,
    ]
) -> List[str]:
    trait_modes = []
    for num_sentence in num_sentences:
        for trait_type in ALL_DIMENSION_NAMES:
            trait_modes.append(f"{trait_type}-{num_sentence}")
    return trait_modes


class TraitGenerator:
    """
    A class to generate student traits from example conversations for multi-turn scenarios.

    This class handles the generation of different types of student traits
    based on conversation analysis, including general behavior, cognitive
    understanding, learning style, and combined descriptions.
    """

    def __init__(self, model: ModelWrapper):
        """
        Initialize the TraitGenerator with a model wrapper.

        Args:
            model: ModelWrapper instance for generating traits
        """
        self.model = model

    def get_trait_system_prompt(self, trait_type: str, num_sentences: Optional[int] = None) -> str:
        """
        Get the system prompt for trait generation based on the trait type.

        Loads `prompts/benchmark/v6/students/trait_generator/system.txt` and
        substitutes `{dimension_description}` and `{sentence_count_suffix}`.
        """
        from pathlib import Path
        dimension_description = get_dimension_description(trait_type)

        template_path = (
            Path(__file__).parent.parent.parent
            / "prompts" / "benchmark" / "v6" / "students"
            / "trait_generator" / "system.txt"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read().rstrip("\n")

        if num_sentences is None:
            suffix = ""
        elif num_sentences > 1:
            suffix = f" The description should be {num_sentences} sentences long."
        else:
            suffix = " The description should be 1 sentence long."

        return (
            template
            .replace("{dimension_description}", dimension_description)
            .replace("{sentence_count_suffix}", suffix)
        )

    def parse_trait_output(self, raw_output: str) -> tuple[str, str]:
        """
        Parse the raw model output to extract the description and thinking.

        Args:
            raw_output: Raw output from the model

        Returns:
            Tuple of (description, raw_output) where description is the parsed description
            after 'DESCRIPTION:' and raw_output is the full response
        """
        if "DESCRIPTION:" in raw_output:
            # Split on DESCRIPTION: and take everything after it
            parts = raw_output.split("DESCRIPTION:", 1)
            description = parts[1].strip()
        else:
            # If no DESCRIPTION: marker, treat the entire output as the description
            description = raw_output.strip()

        return description, raw_output

    def parse_trait_type(self, trait_type: str) -> tuple[str, Optional[int]]:
        """
        Parse the trait mode to extract trait type and sentence count.

            Args:
                student_simulation_mode: Student simulation mode string (e.g., 'natural_trait-3')

            Returns:
                Tuple of (trait_type, num_sentences) where num_sentences can be None
        """
        if "-" not in trait_type:
            base_trait_type = trait_type
            num_sentences = None
        else:
            base_trait_type, num_sentences = trait_type.split("-")
            num_sentences = int(num_sentences)

        if base_trait_type not in ALL_DIMENSION_NAMES:
            raise ValueError(
                f"Invalid base trait type: {base_trait_type}. Must start with one of: {ALL_DIMENSION_NAMES}"
            )

        if num_sentences is not None:
            if num_sentences <= 0:
                raise ValueError(
                    f"Invalid number of sentences: {num_sentences}. Must be greater than 0."
                )

        return base_trait_type, num_sentences

    def _get_user_prompt(self, conversation_text: str, num_sentences: Optional[int] = None) -> str:
        """
        Get the user prompt for trait generation.

        Loads `prompts/benchmark/v6/students/trait_generator/user.txt` and
        substitutes `{conversation_text}` and `{num_sentences}`.
        """
        from pathlib import Path
        template_path = (
            Path(__file__).parent.parent.parent
            / "prompts" / "benchmark" / "v6" / "students"
            / "trait_generator" / "user.txt"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read().rstrip("\n")
        return (
            template
            .replace("{conversation_text}", conversation_text)
            .replace("{num_sentences}", str(num_sentences))
        )

    def _generate_joined_trait(
        self,
        conversation_text: str,
        num_sentences: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        do_return_message_history: bool = False,
        do_return_raw_output: bool = False,
        individual_dimensions: List[str] = None,
        **generation_kwargs,
    ) -> str:
        """
        Generate a 'joined' trait by generating traits for each dimension separately
        and then consolidating them into a single description.

        Args:
            conversation_text: The conversation text to analyze
            num_sentences: Optional number of sentences for the consolidated description
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for generation
            do_return_message_history: If True, return message history as well
            do_return_raw_output: If True, return raw output along with parsed description

        Returns:
            Consolidated trait description (or tuple with message history/raw output if requested)
        """
        from benchmark.synth_students.dimension import ALL_DIMENSIONS

        # Get all individual dimension names (exclude 'all' and 'joined')
        # if individual_dimensions is not specified, use all dimensions
        if individual_dimensions is None:
            individual_dimensions = list(ALL_DIMENSIONS.keys())

        # Generate trait for each dimension
        dimension_traits = {}
        for dimension in individual_dimensions:
            # Use the same num_sentences for each dimension (or None if not specified)
            dimension_mode = dimension
            if num_sentences is not None:
                dimension_mode = f"{dimension}-{num_sentences}"

            trait = self.generate_trait(
                conversation_text=conversation_text,
                trait_mode=dimension_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                do_return_message_history=False,
                do_return_raw_output=False,
                **generation_kwargs,
            )
            dimension_traits[dimension] = trait

        # Consolidate all traits into a single description
        consolidated_trait = "\n\n".join(list(sorted(dimension_traits.values())))

        # Handle return formats
        if do_return_message_history or do_return_raw_output:
            # TODO: hacky, but set to None because we don't have a message history for joined traits (have separate calls for each dimension)
            raw_output = None
            message_history_str = None

            if do_return_message_history:
                if do_return_raw_output:
                    return consolidated_trait, message_history_str, raw_output
                return consolidated_trait, message_history_str

            if do_return_raw_output:
                return consolidated_trait, raw_output

        return consolidated_trait

    def generate_trait(
        self,
        conversation_text: str,
        trait_mode: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        do_return_message_history: bool = False,
        do_return_raw_output: bool = False,
        **generation_kwargs,
    ) -> str:
        """
        Generate a student trait description from a conversation.

        Args:
            conversation_text: The conversation text to analyze
            trait_mode: Trait mode string
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for generation
            do_return_message_history: If True, return message history as well
            do_return_raw_output: If True, return raw output along with parsed description

        Returns:
            Generated trait description string (parsed to extract DESCRIPTION part)
            If do_return_message_history=True, returns (description, message_history_str)
            If do_return_raw_output=True and do_return_message_history=True,
                returns (description, message_history_str, raw_output)

        Raises:
            ValueError: If trait_mode is not supported
        """
        # Parse the simulation mode
        trait_type, num_sentences = self.parse_trait_type(trait_mode)

        # Validate trait type
        if trait_type not in ALL_DIMENSION_NAMES:
            raise ValueError(
                f"Invalid trait mode: {trait_mode}. Must start with one of: {ALL_DIMENSION_NAMES}"
            )

        # Special handling for 'joined' trait type
        if trait_type in [
            "joined",
            "joined_misconceptions_distractedness",
            "joined_misconceptions_affect",
        ]:
            if trait_type == "joined_misconceptions_distractedness":
                individual_dimensions = ["misconceptions", "distractedness"]
            elif trait_type == "joined_misconceptions_affect":
                individual_dimensions = ["misconceptions", "affect"]
            elif trait_type == "joined":
                individual_dimensions = list(ALL_DIMENSIONS.keys())
            else:
                raise ValueError(
                    f'Invalid trait type: {trait_type}. Must be one of: ["joined", "joined_misconceptions_distractedness", "joined_misconceptions_affect"]'
                )

            return self._generate_joined_trait(
                conversation_text=conversation_text,
                num_sentences=num_sentences,
                temperature=temperature,
                max_tokens=max_tokens,
                do_return_message_history=do_return_message_history,
                do_return_raw_output=do_return_raw_output,
                individual_dimensions=individual_dimensions,
                **generation_kwargs,
            )

        # Get system prompt
        system_prompt = self.get_trait_system_prompt(trait_type, num_sentences)

        # Generate trait
        user_prompt = self._get_user_prompt(conversation_text, num_sentences)
        non_system_messages = [{"role": "user", "content": user_prompt}]
        trait_response = self.model.call(
            non_system_messages=non_system_messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **generation_kwargs,
        )

        # Parse the output to extract description
        description, raw_output = self.parse_trait_output(trait_response)

        if do_return_message_history:
            message_history_str = get_history_str(
                non_system_messages + [{"role": "assistant", "content": raw_output}],
                system_prompt=system_prompt,
            )
            if do_return_raw_output:
                return description, message_history_str, raw_output
            return description, message_history_str

        if do_return_raw_output:
            return description, raw_output

        return description

    def generate_trait_with_metadata(
        self,
        conversation_text: str,
        trait_mode: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **generation_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a student trait description with metadata.

        Args:
            conversation_text: The conversation text to analyze
            trait_mode: Trait mode string
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for generation

        Returns:
            Dictionary containing trait description, raw output, and metadata
        """
        trait_type, num_sentences = self.parse_trait_type(trait_mode)

        trait_description, message_history_str, raw_output = self.generate_trait(
            conversation_text=conversation_text,
            trait_mode=trait_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            do_return_message_history=True,
            do_return_raw_output=True,
            **generation_kwargs,
        )

        return {
            "trait_description": trait_description,
            "raw_output": raw_output,
            "trait_type": trait_type,
            "num_sentences": num_sentences,
            "trait_mode": trait_mode,
            "message_history_str": message_history_str,
        }

    def generate_batch_traits(
        self,
        conversation_texts: List[str],
        trait_mode: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        write_cache_idx: Optional[int] = None,
        print_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate traits for multiple conversations in a batch.

        Args:
            conversation_texts: List of conversation texts to analyze
            trait_mode: Trait mode string
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for generation

        Returns:
            List of trait metadata dictionaries
        """
        # Parse the trait type
        trait_type, num_sentences = self.parse_trait_type(trait_mode)

        if trait_type in ["joined", "joined_misconceptions_distractedness", "joined_misconceptions_affect"]:
            return self._generate_batch_joined_traits(
                conversation_texts=conversation_texts,
                trait_type=trait_type,
                trait_mode=trait_mode,
                num_sentences=num_sentences,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=batch_size,
                write_cache_idx=write_cache_idx,
                print_idx=print_idx,
            )

        system_prompt = self.get_trait_system_prompt(trait_type, num_sentences)

        # Prepare batch messages
        batch_messages = []
        for conversation_text in conversation_texts:
            user_prompt = self._get_user_prompt(conversation_text, num_sentences)
            batch_messages.append([{"role": "user", "content": user_prompt}])

        # Generate traits in batch
        batch_responses = self.model.batch_call(
            non_system_messages_list=batch_messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            batch_size=batch_size,
            write_cache_idx=write_cache_idx,
            print_idx=print_idx,
        )

        # Process responses
        trait_metadata_list = []
        for response, messages in zip(batch_responses, batch_messages):
            # Parse the output to extract description
            description, raw_output = self.parse_trait_output(response)

            message_history_str = get_history_str(
                messages + [{"role": "assistant", "content": raw_output}],
                system_prompt=system_prompt,
            )
            trait_metadata_list.append(
                {
                    "trait_description": description,
                    "raw_output": raw_output,
                    "trait_type": trait_type,
                    "num_sentences": num_sentences,
                    "trait_mode": trait_mode,
                    "system_prompt": system_prompt,
                    "message_history_str": message_history_str,
                }
            )

        return trait_metadata_list

    def _generate_batch_joined_traits(
        self,
        conversation_texts: List[str],
        trait_type: str,
        trait_mode: str,
        num_sentences: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        write_cache_idx: Optional[int] = None,
        print_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Batch version of _generate_joined_trait: for each individual dimension,
        batch-extract traits across all conversations, then consolidate per conversation."""
        from benchmark.synth_students.dimension import ALL_DIMENSIONS

        if trait_type == "joined_misconceptions_distractedness":
            individual_dimensions = ["misconceptions", "distractedness"]
        elif trait_type == "joined_misconceptions_affect":
            individual_dimensions = ["misconceptions", "affect"]
        else:
            individual_dimensions = list(ALL_DIMENSIONS.keys())

        # {dimension: [trait_description_for_conv_0, trait_description_for_conv_1, ...]}
        dimension_results: Dict[str, List[str]] = {}

        for dimension in individual_dimensions:
            dim_mode = f"{dimension}-{num_sentences}" if num_sentences is not None else dimension
            dim_metadata = self.generate_batch_traits(
                conversation_texts=conversation_texts,
                trait_mode=dim_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=batch_size,
                write_cache_idx=write_cache_idx,
                print_idx=print_idx,
            )
            dimension_results[dimension] = [m["trait_description"] for m in dim_metadata]

        trait_metadata_list = []
        for i in range(len(conversation_texts)):
            consolidated = "\n\n".join(
                sorted(dimension_results[dim][i] for dim in individual_dimensions)
            )
            trait_metadata_list.append(
                {
                    "trait_description": consolidated,
                    "raw_output": None,
                    "trait_type": trait_type,
                    "num_sentences": num_sentences,
                    "trait_mode": trait_mode,
                    "system_prompt": None,
                    "message_history_str": None,
                }
            )

        return trait_metadata_list


class BaseTraitEditor:
    """
    Base class for trait editors with shared functionality.
    """

    def __init__(self, model: ModelWrapper):
        """
        Initialize the BaseTraitEditor with a model wrapper.

        Args:
            model: ModelWrapper instance for editing traits
        """
        self.model = model


class DistractorTraitEditor(BaseTraitEditor):
    """
    A class to edit student traits by incorporating traits from a distractor trait
    along a specific dimension while preserving other aspects of the original trait.

    This is useful for creating "hard distractors" in trait matching evaluations,
    where the distractor is similar to the original but differs along a specific
    trait dimension (e.g., misconception_speed, learning_speed).
    """

    def __init__(self, model: ModelWrapper):
        """
        Initialize the DistractorTraitEditor with a model wrapper.

        Args:
            model: ModelWrapper instance for editing traits
        """
        super().__init__(model)

    def edit_trait_with_distractor_traits(
        self,
        original_trait: str,
        distractor_trait: str,
        trait_dimension: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        num_sentences: Optional[int] = None,
        **generation_kwargs,
    ) -> str:
        """
        Edit the original trait to incorporate traits from the distractor trait
        along a specific dimension, while preserving all other aspects of the original.

        Args:
            original_trait: The original trait description to edit
            distractor_trait: The distractor trait to extract traits from
            trait_dimension: The dimension along which to incorporate distractor traits
                              (e.g., 'misconception_speed', 'natural_trait_learning_speed')
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for generation

        Returns:
            Edited trait description string

        Raises:
            ValueError: If trait_dimension is not recognized
        """

        dimension_description = get_dimension_description(trait_dimension)

        system_prompt = """You are an expert at editing student trait descriptions. Your task is to edit a trait description by incorporating specific traits from another trait along a particular dimension, while keeping everything else exactly the same.

You will be given:
1. An original trait description
2. A distractor trait description
3. A specific dimension to focus on

Your task is to:
- Extract traits from the distractor trait that relate to the specified dimension
- Edit the original trait to incorporate ONLY those traits along that dimension
- Keep ALL other aspects of the original trait unchanged (e.g., conversation topics, other behavioral traits, general structure)
- Maintain the same writing style and level of detail as the original
- Do not add new information beyond what's in the distractor trait for that dimension"""

        user_prompt = f"""Original Trait:
{original_trait}

Distractor Trait:
{distractor_trait}

Dimension to : {dimension_description}

Please edit the original trait to incorporate traits from the distractor trait along the specified dimension, while keeping everything else the same."""
        if num_sentences is not None:
            user_prompt += f"""The edited trait should be {num_sentences} sentences long."""

        edited_trait = self.model.call(
            non_system_messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **generation_kwargs,
        )

        return edited_trait

    def edit_trait_with_metadata(
        self,
        original_trait: str,
        distractor_trait: str,
        trait_dimension: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        num_sentences: Optional[int] = None,
        **generation_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Edit a trait with distractor traits and return metadata.

        Args:
            original_trait: The original trait description to edit
            distractor_trait: The distractor trait to extract traits from
            trait_dimension: The dimension along which to incorporate distractor traits
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for generation

        Returns:
            Dictionary containing edited trait and metadata
        """
        edited_trait = self.edit_trait_with_distractor_traits(
            original_trait=original_trait,
            distractor_trait=distractor_trait,
            trait_dimension=trait_dimension,
            num_sentences=num_sentences,
            temperature=temperature,
            max_tokens=max_tokens,
            **generation_kwargs,
        )

        return {
            "edited_trait": edited_trait,
            "original_trait": original_trait,
            "distractor_trait": distractor_trait,
            "trait_dimension": trait_dimension,
            "temperature": temperature,
        }


class UnconditionalTraitEditor(BaseTraitEditor):
    """
    A class to edit student traits to be different along a specific dimension
    without requiring a distractor trait.

    This creates edited traits by asking an LLM to modify the original trait
    to be different along a specified dimension, while preserving all other aspects.
    This is useful for creating distractors when you don't have a reference distractor trait.
    """

    def __init__(self, model: ModelWrapper):
        """
        Initialize the UnconditionalTraitEditor with a model wrapper.

        Args:
            model: ModelWrapper instance for editing traits
        """
        super().__init__(model)

    def edit_trait_unconditionally(
        self,
        original_trait: str,
        trait_dimension: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        num_sentences: Optional[int] = None,
        **generation_kwargs,
    ) -> str:
        """
        Edit the original trait to be different along a specific dimension,
        while preserving all other aspects of the original.

        Args:
            original_trait: The original trait description to edit
            trait_dimension: The dimension along which to make the trait different
                              (e.g., 'misconception_speed', 'natural_trait_learning_speed')
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for generation
            num_sentences: Optional number of sentences for the edited trait

        Returns:
            Edited trait description string

        Raises:
            ValueError: If trait_dimension is not recognized
        """
        dimension_description = get_dimension_description(trait_dimension)

        system_prompt = """You are an expert at editing student persona descriptions. Your task is to edit a persona description to be different along a particular dimension, while keeping everything else exactly the same.

You will be given:
1. An original persona description
2. A specific dimension to focus on

Your task is to:
- Edit the original persona to be DIFFERENT along the specified dimension
- Keep ALL other aspects of the original persona unchanged (e.g., conversation topics, other behavioral traits, general structure)
- Maintain the same writing style and level of detail as the original
- Make the change meaningful but realistic - the edited persona should represent a different student characteristic along that dimension"""

        user_prompt = f"""Original Persona:
{original_trait}

Dimension to modify: {dimension_description}

Please edit the original persona to be different along the specified dimension, while keeping everything else the same."""

        if num_sentences is not None:
            if num_sentences > 1:
                user_prompt += f"\n\nThe edited persona should be {num_sentences} sentences long."
            else:
                user_prompt += f"\n\nThe edited persona should be 1 sentence long."

        edited_trait = self.model.call(
            non_system_messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **generation_kwargs,
        )

        return edited_trait

    def edit_trait_unconditionally_with_metadata(
        self,
        original_trait: str,
        trait_dimension: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        num_sentences: Optional[int] = None,
        **generation_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Edit a trait unconditionally and return metadata.

        Args:
            original_trait: The original trait description to edit
            trait_dimension: The dimension along which to make the trait different
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for generation
            num_sentences: Optional number of sentences for the edited trait

        Returns:
            Dictionary containing edited trait and metadata
        """
        edited_trait = self.edit_trait_unconditionally(
            original_trait=original_trait,
            trait_dimension=trait_dimension,
            temperature=temperature,
            max_tokens=max_tokens,
            num_sentences=num_sentences,
            **generation_kwargs,
        )

        return {
            "edited_trait": edited_trait,
            "original_trait": original_trait,
            "trait_dimension": trait_dimension,
            "temperature": temperature,
            "num_sentences": num_sentences,
        }
