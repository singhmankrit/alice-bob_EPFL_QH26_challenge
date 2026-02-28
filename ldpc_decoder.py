
class LDPC_Sinter_Wrapper(sinter.Decoder):
    def __init__(self, osd_order=0, osd_method="osd_cs"):
        self.osd_order = osd_order
        self.osd_method = osd_method

    def decode_via_files(self, *, num_detectors, num_observables, 
                        dem_path, shots_path, stats_path, tmp_dir):
        # 1. Load the Detector Error Model
        with open(dem_path) as f:
            dem = stim.DetectorErrorModel(f.read())
            
        # 2. Extract the H matrix and error probabilities
        # We use pymatching's helper because it's the fastest way to get H from DEM
        m = pymatching.Matching.from_detector_error_model(dem)
        H = m.check_matrix
        
        # 3. Initialize the BPOSD decoder
        # If osd_method is None, this performs "Pure BP"
        decoder = bposd_decoder(
            H, 
            error_rate=0.01, # Starting guess for BP
            max_iter=H.shape[1], 
            bp_method="log_domain",
            osd_method=self.osd_method, 
            osd_order=self.osd_order
        )

        # 4. Use sinter's built-in helper to run the decoding loop
        # This handles the shots/stats files automatically for you
        return sinter.anon_decode_helper(
            num_detectors=num_detectors,
            num_observables=num_observables,
            shots_path=shots_path,
            stats_path=stats_path,
            decoder=lambda syndrome: decoder.decode(syndrome)
        )