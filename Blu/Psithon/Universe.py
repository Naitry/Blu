# Typing
from typing import Optional

# Compute
import torch

# Blu
from Blu.Psithon.Fields.Field import (Field,
                                      BLU_PSITHON_defaultDataType)
from Blu.Psithon.Particles.ParticleCloud import ParticleCloud


class Universe:
    def __init__(self,
                 device: torch.device,
                 spatialDimensions: int,
                 resolution: int,
                 scale: float,
                 speedLimit: float,
                 dt: float,
                 delta: float,
                 fields: Optional[list[Field]] = None,
                 particles: Optional[list[ParticleCloud]] = None,
                 dtype: torch.dtype = BLU_PSITHON_defaultDataType):
        """
        Universe constructor

        :param device: the torch capable device which the universe will be created on
        :param spatialDimensions: number of spatial dimensions in the Universe
        :param resolution: the resolution of the fields in each dimension
        :param scale: describes the total scale of the universe in real world units
        :param speedLimit: the maximum speed of propagation in the universe
        :param dt: the regular magnitude of the forward timestep for the universe
        :param delta: the magnitude of the distance between points in the system, heavily effects stability
        :param fields: an optional set of fields, if input this will be the basis of the Universe
        :param particles: an optional set of particles
        :param dtype: the torch data type of the tensors
        """
        self.spatialDimensions: int = spatialDimensions
        self.resolution: int = resolution
        self.scale: float = scale
        self.c: float = speedLimit
        self.dt: float = dt
        self.delta: float = delta
        self.fields: list[Field] = fields or []
        self.particles: list[ParticleCloud] = particles or []
        self.particles: list[ParticleCloud] = particles or []
        self.fields: list[Field] = fields or []
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

    def addField(self,
                 name: str) -> None:
        """
        add a new field to the Universe

        :param name: the string name of the field

        :return: none
        """
        self.fields.append(Field(name=name,
                                 device=self.device,
                                 dtype=self.dtype,
                                 spatialDimensions=self.spatialDimensions,
                                 resolution=self.resolution))

    def addParticleCloud(self,
                         name: str) -> None:
        """
        add a new field to the Universe

        :param name: the string name of the field

        :return: none
        """
        self.fields.append(ParticleCloud(name=name,
                                         device=self.device,
                                         dtype=self.dtype,
                                         spatialDimensions=self.spatialDimensions,
                                         resolution=self.resolution))

    def update(self,
               dt: Optional[float] = None,
               delta: Optional[float] = None) -> None:

        self.updateParticleClouds(dt=dt or self.dt,
                                  delta=delta or self.delta)
        self.updateFields(dt=dt or self.dt,
                          delta=delta or self.delta)

    def updateFields(self,
                     dt: Optional[float] = None,
                     delta: Optional[float] = None) -> None:
        """
        update each field in the universe

        :param dt: the magnitude of the timestep forward which will be taken
        :param delta: the magnitude of the distance between points in the system, heavily effects stability
        :param device: the device which the calculation will be made on
        this should usually be default, to keep the data on one device

        :return: none
        """
        for field in self.fields:
            field.update(dt=dt or self.dt,
                         delta=delta or self.delta)

    def updateParticleClouds(self,
                             dt: Optional[float] = None,
                             delta: Optional[float] = None) -> None:
        """
        update each field in the universe

        :param dt: the magnitude of the timestep forward which will be taken
        :param delta: the magnitude of the distance between points in the system, heavily effects stability
        :param device: the device which the calculation will be made on
        this should usually be default, to keep the data on one device

        :return: none
        """
        for cloud in self.particles:
            cloud.update(dt=dt or self.dt,
                         delta=delta or self.delta)
