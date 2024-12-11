"""Module is the base class for all modules in Minitorch."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes
    ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        """Initialize the module."""
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module.

        Returns
        -------
            Sequence[Module]: The direct child modules of this module.

        """
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the mode of this module and all descendent modules to `train` by calling recursively."""
        self.training = True
        # calling recursively to set the mode of all descendent modules to `train`
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        """Set the mode of this module and all descendent modules to `eval`."""
        self.training = False
        # calling recursively to set the mode of all descendent modules to `eval`
        for module in self.modules():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Returns
        -------
            Sequence[Tuple[str, Parameter]]: The name and `Parameter` of each ancestor parameter in the format
            'ancestor_module_name.parameter_name'.

        """

        def collect_parameters(module: Module, prefix: str) -> None:
            """Collects all the parameters of the module and its descendents.

            Args:
            ----
                module (Module): The module to collect parameters from.
                prefix (str): The prefix of the module.

            """
            for name, param in module._parameters.items():
                full_name = f"{prefix}.{name}" if prefix else name
                params.append((full_name, param))
            # Recursively collect parameters from child modules
            for child_name, child_module in module._modules.items():
                collect_parameters(
                    child_module,
                    child_name if prefix == "" else f"{prefix}.{child_name}",
                )

        params = []
        # Start collecting from the current module
        collect_parameters(self, "")
        return params

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents.

        Returns
        -------
            Sequence[Parameter]: A list of all the parameters of this module and its descendents.

        """
        params = []
        for _, param in self.named_parameters():
            params.append(param)
        return params

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k (str): Local name of the parameter.
            v (Any): Value for the parameter.

        Returns:
        -------
            Parameter: The newly created parameter.

        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        """Set an attribute of the module.

        Args:
        ----
            key (str): The name of the attribute to set.
            val (Parameter): The value to set the attribute to.

        """
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        """Get an attribute of the module.

        Args:
        ----
            key (str): The name of the attribute to get.

        Returns:
        -------
            Any: The value of the attribute.

        """
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the module.

        Args:
        ----
            *args: The arguments to pass to the module.
            **kwargs: The keyword arguments to pass to the module.

        Returns:
        -------
            Any: The output of the module.

        """
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """Return the string representation of the module.

        Returns
        -------
            str: The string representation of the module.

        """

        def _addindent(s_: str, numSpaces: int) -> str:
            """Add indentation to the string.

            Args:
            ----
                s_ (str): The string to add indentation to.
                numSpaces (int): The number of spaces to add.

            Returns:
            -------
                str: The string with indentation.

            """
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        """Initialize the parameter.

        Args:
        ----
            x (Any): The value to store in the parameter.
            name (Optional[str]): The name of the parameter.

        """
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value.

        Args:
        ----
            x (Any): The new value to store in the parameter.

        """
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        """Return the string representation of the parameter.

        Returns
        -------
            str: The string representation of the parameter.

        """
        return repr(self.value)

    def __str__(self) -> str:
        """Return the string representation of the parameter.

        Returns
        -------
            str: The string representation of the parameter.

        """
        return str(self.value)
