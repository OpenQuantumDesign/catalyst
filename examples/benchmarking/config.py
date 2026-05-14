from artiq.experiment import *
from artiq.coredevice.core import (
    Core,
)
from artiq.master.worker_impl import Scheduler
from artiq.coredevice.ttl import TTLOut, TTLInOut
from artiq.coredevice.ad9912 import AD9912

import math

########################################################################################

pi = math.pi
s = 1.0
ms = 1e-3
us = 1e-6
ns = 1e-9

Hz = 1.0
kHz = 1e3
MHz = 1e6
GHz = 1e9

dB = 1.0

########################################################################################


class AD9912_DDS(AD9912):
    sw: TTLOut


class AD9910_DDS(AD9912):
    sw: TTLOut


########################################################################################


class Bloodstone_Experiment(HasEnvironment):
    core: Core
    scheduler: Scheduler

    image_pmt_Counter: TTLInOut
    detection_aom_DDS: AD9910_DDS
    cooling_aom1_DDS: AD9910_DDS
    cooling_aom2_DDS: AD9910_DDS
    pumping_aom_DDS: AD9910_DDS

    RMN_PLL: TTLOut
    AWG_TTL: TTLOut

    ttl0: TTLInOut
    urukul0_ch0: AD9910_DDS
    urukul0_ch1: AD9910_DDS
    urukul0_ch2: AD9910_DDS
    urukul0_ch3: AD9910_DDS
    ttl4: TTLOut
    ttl7: TTLOut
    ttl6: TTLOut

    def __init__(self, managers_or_parent, *args, **kwargs):
        self.mapping_dict = {
            "core": "core",
            "image_pmt_Counter": "ttl0",
            "detection_aom_DDS": "urukul0_ch0",
            "cooling_aom1_DDS": "urukul0_ch1",
            "cooling_aom2_DDS": "urukul0_ch2",
            "pumping_aom_DDS": "urukul0_ch3",
            "RMN_PLL": "ttl7",
            "ttl6": "ttl6",
            "AWG_TTL": "ttl4",
            "awg": "awg",
        }

        if managers_or_parent:
            super().__init__(managers_or_parent, *args, **kwargs)
        else:
            print(
                "if you see this print it means that the device_db is being loaded.\n This should appear only once when you open the master. If you see this message twice then there must be an error somewhere.\n To debug, please remove the whole __init__ function. and remove the last few lines of the device_db."
            )

    def build(self):
        self.init()

    def init(self):
        """Specify devices used in the experiment."""

        for key, value in self.mapping_dict.items():
            self.setattr_device(value)
            setattr(self, key, self.get_device(value))
            self.update_kernel_invariants(key)

    def update_kernel_invariants(self, key):
        kernel_invariants = getattr(self, "kernel_invariants", set())
        self.kernel_invariants = kernel_invariants | {key}
