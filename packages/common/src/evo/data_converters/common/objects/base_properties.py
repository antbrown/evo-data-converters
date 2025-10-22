import typing
from abc import ABC, abstractmethod


class BaseObjectProperties(ABC):
    def __init__(
        self,
        *,
        name: str,
        uuid: str | None = None,
        description: str | None = None,
        extensions: dict[str, typing.Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.name: str = name
        self.uuid: str | None = uuid
        self.description: str | None = description
        self.extensions: dict[str, typing.Any] | None = extensions
        self.tags: dict[str, str] | None = tags


class BaseSpatialDataProperties(BaseObjectProperties):
    def __init__(
        self,
        *,
        name: str,
        uuid: str | None = None,
        coordinate_reference_system: int | str | None = None,
        description: str | None = None,
        extensions: dict[str, typing.Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name, uuid=uuid, description=description, extensions=extensions, tags=tags)
        self.coordinate_reference_system: int | str | None = coordinate_reference_system

    def get_bounding_box(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
