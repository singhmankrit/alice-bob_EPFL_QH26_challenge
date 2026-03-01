import numpy as np
import sinter
import stim
from ldpc import BpOsdDecoder  # v2 API
from typing import Optional


class LDPC_CompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, decoder, num_detectors: int, observables: np.ndarray):
        self._decoder = decoder
        self._num_detectors = num_detectors
        self._observables = observables

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
        bit_packed_observable_data: np.ndarray = None,  # make optional
        **kwargs,
    ) -> np.ndarray:
        num_shots = bit_packed_detection_event_data.shape[0]
        num_observables = self._observables.shape[1]

        syndromes = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            count=self._num_detectors,
            bitorder="little",
        )

        num_obs_bytes = (num_observables + 7) // 8
        result = np.zeros((num_shots, num_obs_bytes), dtype=np.uint8)

        for i in range(num_shots):
            correction = self._decoder.decode(syndromes[i].astype(np.uint8))
            obs_bits = (self._observables.T @ correction) % 2
            result[i] = np.packbits(obs_bits, bitorder="little")[:num_obs_bytes]

        return result


class LDPC_Sinter_Decoder(sinter.Decoder):
    def __init__(self, osd_order: int = 0, osd_method: Optional[str] = "osd_cs"):
        self.osd_order = osd_order
        self.osd_method = osd_method

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
        **kwargs,
    ) -> sinter.CompiledDecoder:
        H, observables = dem_to_parity_check_matrix(dem)
        num_detectors = H.shape[0]

        decoder = BpOsdDecoder(
            H,
            error_rate=0.01,
            max_iter=num_detectors,
            bp_method="product_sum",
            osd_method=self.osd_method if self.osd_method is not None else "osd_0",
            osd_order=self.osd_order,
        )

        return LDPC_CompiledDecoder(decoder, num_detectors, observables)


def dem_to_parity_check_matrix(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray]:
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables

    detector_lists = []
    observable_lists = []

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets = []
            obs = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    dets.append(target.val)
                elif target.is_logical_observable_id():
                    obs.append(target.val)
            detector_lists.append(dets)
            observable_lists.append(obs)

    num_errors = len(detector_lists)
    H = np.zeros((num_detectors, num_errors), dtype=np.uint8)
    observables = np.zeros((num_errors, num_observables), dtype=np.uint8)

    for col, (dets, obs) in enumerate(zip(detector_lists, observable_lists)):
        for d in dets:
            H[d, col] = 1
        for o in obs:
            observables[col, o] = 1

    return H, observables