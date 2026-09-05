"""Push-based diagnostic sensors for local Groq request usage."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.const import EntityCategory, PERCENTAGE, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_MODEL,
    CONF_NAME,
    CONF_SUBENTRY_ID,
    FEATURE_TEXT_GENERATION,
    FEATURE_IMAGE_RECOGNITION,
    UNIQUE_ID,
)
from .entity import service_device_info
from .runtime import async_get_runtime
from .types import GroqConfigEntry
from .usage import GroqUsage

PARALLEL_UPDATES = 0

SENSORS = (
    SensorEntityDescription(
        key="requests",
        translation_key="requests",
        native_unit_of_measurement="requests",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    *(
        SensorEntityDescription(
            key=key,
            translation_key=key,
            native_unit_of_measurement="tokens",
            state_class=SensorStateClass.MEASUREMENT,
        )
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
        )
    ),
    SensorEntityDescription(
        key="response_time",
        translation_key="response_time",
        native_unit_of_measurement=UnitOfTime.SECONDS,
        device_class=SensorDeviceClass.DURATION,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="cache_hit_rate",
        translation_key="cache_hit_rate",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add disabled-by-default measurements to existing text and vision devices."""
    runtime = await async_get_runtime(hass, entry)
    for service_type in (FEATURE_TEXT_GENERATION, FEATURE_IMAGE_RECOGNITION):
        for service in runtime.services_by_type.get(service_type, ()):
            async_add_entities(
                [
                    GroqUsageSensor(service, runtime.client.usage, description)
                    for description in SENSORS
                ],
                config_subentry_id=service.get(CONF_SUBENTRY_ID),
            )


class GroqUsageSensor(SensorEntity):
    """A latest-request measurement, scoped to one configured Groq service."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_entity_registry_enabled_default = False

    def __init__(
        self,
        service: dict[str, Any],
        usage: GroqUsage,
        description: SensorEntityDescription,
    ) -> None:
        self.entity_description = description
        self._service = service
        self._service_id = str(service[UNIQUE_ID])
        self._usage = usage
        self._attr_unique_id = f"{self._service_id}_{description.key}"

    @property
    def device_info(self) -> DeviceInfo:
        return service_device_info(
            self._service_id, self._service[CONF_MODEL], self._service[CONF_NAME]
        )

    @property
    def native_value(self) -> float | None:
        return self._usage.values.get(self._service_id, {}).get(
            self.entity_description.key
        )

    async def async_added_to_hass(self) -> None:
        """Stop updates automatically on entity removal or entry unload."""
        self.async_on_remove(self._usage.subscribe(self._usage_updated))

    @callback
    def _usage_updated(self) -> None:
        self.async_write_ha_state()
