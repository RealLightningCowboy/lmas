from __future__ import annotations

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox


class _DeferredNumericMixin:
    """Make spin-box text behave like an ordinary editable numeric field.

    Qt normally normalizes spin-box text aggressively.  LMAS permits blank and
    partial text while the user is editing, commits only on Enter/focus-out,
    and restores the previous valid value when the committed text is invalid.
    """

    _partial_tokens = {"", "+", "-", ".", "+.", "-."}

    def _init_deferred_numeric(self) -> None:
        self.setKeyboardTracking(False)
        self._lmas_previous_value = self.value()
        self.lineEdit().selectionChanged.connect(self._remember_value)
        self.editingFinished.connect(self._restore_invalid_text)

    def _remember_value(self) -> None:
        if self.hasFocus():
            self._lmas_previous_value = self.value()

    def _numeric_text(self, text: str) -> str:
        value = str(text).strip()
        prefix = str(self.prefix())
        suffix = str(self.suffix())
        if prefix and value.startswith(prefix):
            value = value[len(prefix):]
        if suffix and value.endswith(suffix):
            value = value[:-len(suffix)]
        return value.strip()

    def validate(self, text: str, pos: int):  # noqa: N802 - Qt API
        if self._numeric_text(text) in self._partial_tokens:
            return QValidator.State.Intermediate, text, pos
        return super().validate(text, pos)

    def _restore_invalid_text(self) -> None:
        text = self._numeric_text(self.lineEdit().text())
        state, _text, _pos = self.validate(self.lineEdit().text(), 0)
        if text in self._partial_tokens or state == QValidator.State.Invalid:
            previous = self._lmas_previous_value
            blocked = self.blockSignals(True)
            try:
                self.setValue(previous)
                self.lineEdit().setText(self.textFromValue(previous) + self.suffix())
            finally:
                self.blockSignals(blocked)
        else:
            self._lmas_previous_value = self.value()


class DeferredDoubleSpinBox(_DeferredNumericMixin, QDoubleSpinBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_deferred_numeric()

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API
        text = f"{float(value):.{self.decimals()}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in {"-0", "+0", ""} else text

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt API
        core = self._numeric_text(text)
        if core in self._partial_tokens:
            return float(self._lmas_previous_value)
        return super().valueFromText(text)


class DeferredSpinBox(_DeferredNumericMixin, QSpinBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_deferred_numeric()

    def valueFromText(self, text: str) -> int:  # noqa: N802 - Qt API
        core = self._numeric_text(text)
        if core in self._partial_tokens:
            return int(self._lmas_previous_value)
        return super().valueFromText(text)


__all__ = ["DeferredDoubleSpinBox", "DeferredSpinBox"]
