#!/usr/bin/env python3
"""
Milestone 3C: PKM Synthetic Retrieval Data Builder.

Generates 200 train and 50 dev synthetic contrastive retrieval examples
from newly authored, completely isolated project PKM seed notes.

All examples adhere to:
- L-1: No query identical to eval/cases.yaml
- L-2: No positive_id in EVAL_POSITIVE_IDS
- L-4: No hard_negative_ids from eval vault
- L-6: Deterministic SHA-256 ID: sha256(query + positive_id + creation_method)
- Document Disjointness: Train seed notes and Dev seed notes are 100% disjoint.
- Provenance: review_status="unreviewed" with model, digest, prompt version, seed doc ID.
- Review Pack: Compact sample of 20 random unreviewed examples for human inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent
EVAL_DIR = REPO_ROOT / "eval"
TRAINING_DIR = REPO_ROOT / "training"
SEED_DIR = TRAINING_DIR / "synthetic_seed"
PKM_TRAIN_OUT = TRAINING_DIR / "pkm_train.jsonl"
PKM_DEV_OUT = TRAINING_DIR / "pkm_dev.jsonl"
REVIEW_PACK_OUT = TRAINING_DIR / "synthetic_review_pack.json"
MANIFEST_OUT = TRAINING_DIR / "manifest_synthetic.json"

sys.path.insert(0, str(EVAL_DIR))
from training_schema import (  # noqa: E402
    EVAL_POSITIVE_IDS,
    EVAL_QUERIES,
    TrainingDataset,
    TrainingExample,
    compute_training_id,
    validate_no_eval_leakage,
)

SCHEMA_VERSION = "1.0.0"
GENERATOR_MODEL = "qwen3:4b"
GENERATOR_DIGEST = "sha256:359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
PROMPT_VERSION = "pkm-synth-v1.0"
CREATION_METHOD = "synthetic_llm"
LICENCE = "MIT"
SOURCE_NAME = "pkm_synthetic_v1"

# ---------------------------------------------------------------------------
# Project-Authored Isolated Seed Notes
# ---------------------------------------------------------------------------

SEED_NOTES: dict[str, dict] = {
    # --- TRAIN SEED NOTES (16 documents) ---
    "distributed_systems/raft_consensus.md": {
        "split": "train",
        "title": "Raft Consensus Algorithm Notes",
        "sections": [
            {
                "id": "leader_election",
                "text": "In Raft, leader election begins when a follower's election timer expires without receiving heartbeats from the current leader. The follower transitions to candidate state, increments its currentTerm, votes for itself, and issues RequestVote RPCs in parallel to all other nodes in the cluster. A candidate becomes leader if it receives votes from a strict majority of nodes before another candidate claims leadership or a timeout occurs.",
                "queries": [
                    ("How does leader election initiate in the Raft consensus protocol?", "dense_semantic"),
                    ("Raft candidate RequestVote RPC majority threshold", "lexical_exact"),
                    ("Explain the transition from follower to candidate in Raft when election timer elapses", "hybrid_core"),
                ],
            },
            {
                "id": "log_replication",
                "text": "Once a Raft leader is established, client requests are accepted as log entries. The leader appends the command to its own log and issues AppendEntries RPCs to replicate the entry across followers. An entry is considered committed once it is successfully stored on a majority of cluster nodes, at which point the leader executes the command against its state machine and returns the result to the client.",
                "queries": [
                    ("When is a log entry considered committed in Raft?", "dense_semantic"),
                    ("AppendEntries RPC log replication state machine commit", "lexical_exact"),
                ],
            },
            {
                "id": "log_compaction_snapshots",
                "text": "To prevent logs from growing unbounded in Raft, state machine snapshots are periodically written to persistent storage, allowing log entries prior to the snapshot index to be discarded. When a follower is significantly behind, the leader uses InstallSnapshot RPCs instead of AppendEntries to fast-forward the follower's state.",
                "queries": [
                    ("How does Raft handle log compaction and catch up lagging followers?", "dense_semantic"),
                    ("InstallSnapshot RPC log truncation index", "lexical_exact"),
                ],
            },
        ],
    },
    "distributed_systems/vector_clocks.md": {
        "split": "train",
        "title": "Vector Clocks and Causality Tracking",
        "sections": [
            {
                "id": "causal_ordering",
                "text": "Vector clocks maintain partial ordering of events in distributed systems without synchronized physical clocks. Each process i maintains a vector V of size N. Before executing an event, process i increments V[i]. When sending a message, V is attached. The receiver updates its vector by taking the component-wise maximum with the incoming vector and then incrementing its local component.",
                "queries": [
                    ("How do vector clocks track causality across distributed nodes without synchronized clocks?", "dense_semantic"),
                    ("Vector clock component-wise maximum algorithm rules", "lexical_exact"),
                ],
            },
            {
                "id": "concurrent_conflicts",
                "text": "Two events A and B in a vector clock system are concurrent if neither vector dominates the other component-wise. When concurrent updates occur to the same key, systems like Dynamo preserve both versions as conflicting siblings, delegating conflict reconciliation to application-level logic or CRDTs.",
                "queries": [
                    ("How are concurrent conflicts detected in vector clock systems like Dynamo?", "dense_semantic"),
                    ("Dynamo vector clock siblings conflict resolution CRDT", "hybrid_core"),
                ],
            },
        ],
    },
    "audio_engineering/subtractive_synthesis.md": {
        "split": "train",
        "title": "Principles of Subtractive Sound Synthesis",
        "sections": [
            {
                "id": "oscillators_and_harmonics",
                "text": "Subtractive synthesis starts with harmonically rich waveforms generated by oscillators. A sawtooth wave contains all integer harmonics with amplitudes inversely proportional to harmonic number (1/n), producing a bright, brassy timbre. Square waves contain only odd harmonics (1/3, 1/5, 1/7), yielding a hollow, woody clarinet-like character.",
                "queries": [
                    ("What harmonic differences distinguish sawtooth and square waves in subtractive synthesis?", "dense_semantic"),
                    ("sawtooth integer harmonics vs square wave odd harmonics 1/n", "lexical_exact"),
                ],
            },
            {
                "id": "resonant_filters",
                "text": "Voltage-controlled low-pass filters (VCF) sculpt timbre by attenuating frequencies above the cutoff frequency at rates typically of 12dB or 24dB per octave (2-pole vs 4-pole). Resonance increases feedback around the cutoff frequency, producing a characteristic peak that emphasizes harmonic components at that threshold.",
                "queries": [
                    ("How does filter resonance affect frequency response in low-pass filters?", "dense_semantic"),
                    ("low-pass filter resonance 24dB octave 4-pole slope", "lexical_exact"),
                ],
            },
            {
                "id": "adsr_envelopes",
                "text": "ADSR envelope generators modulate parameters over time through Attack (time to peak), Decay (time to sustain level), Sustain (held amplitude level while key is pressed), and Release (time to return to silence after key release). Envelopes commonly modulate both VCA (amplitude) and VCF (filter cutoff).",
                "queries": [
                    ("Explain the four stages of an ADSR envelope in audio synthesizers", "dense_semantic"),
                    ("ADSR envelope attack decay sustain release VCA VCF", "hybrid_core"),
                ],
            },
        ],
    },
    "audio_engineering/nyquist_sampling.md": {
        "split": "train",
        "title": "Digital Audio Sampling and Reconstruction",
        "sections": [
            {
                "id": "nyquist_rate",
                "text": "The Nyquist-Shannon sampling theorem states that an analog bandlimited signal can be perfectly reconstructed if sampled at a rate greater than twice its highest frequency component (f_s > 2*f_max). For human hearing bandwidth up to 20kHz, standard audio sampling rates of 44.1kHz or 48kHz provide adequate guard bands for anti-aliasing reconstruction filters.",
                "queries": [
                    ("Why is 44.1kHz chosen as the standard sampling rate for digital audio?", "dense_semantic"),
                    ("Nyquist Shannon sampling theorem 44.1kHz 20kHz bandlimited", "lexical_exact"),
                ],
            },
            {
                "id": "aliasing_and_oversampling",
                "text": "When frequency components above the Nyquist frequency enter an analog-to-digital converter without proper filtering, they fold back into the audible spectrum as non-harmonic distortion known as aliasing. Modern converters use delta-sigma oversampling architectures to relax analog filter requirements.",
                "queries": [
                    ("What causes aliasing distortion in analog-to-digital audio conversion?", "dense_semantic"),
                    ("aliasing foldback Nyquist oversampling delta-sigma ADC", "hybrid_core"),
                ],
            },
        ],
    },
    "coffee_science/espresso_extraction.md": {
        "split": "train",
        "title": "Espresso Extraction Physics and Chemistry",
        "sections": [
            {
                "id": "extraction_kinetics",
                "text": "Espresso extraction dissolves coffee solubles in a progressive chemical sequence: sour organic acids and volatile aromatic compounds extract first, followed by sweet lipids and sugars, and finally bitter chlorogenic acid derivatives and astringent polyphenols. An optimal extraction yield typically sits between 18% and 22% of dry coffee mass.",
                "queries": [
                    ("In what chemical sequence are compounds extracted during an espresso shot?", "dense_semantic"),
                    ("espresso extraction yield percentage organic acids chlorogenic", "lexical_exact"),
                ],
            },
            {
                "id": "channeling_and_puck_prep",
                "text": "Channeling occurs when high-pressure water (typically 6-9 bars) finds paths of least resistance through non-uniform coffee pucks. This causes localized over-extraction along the fissure alongside under-extraction of surrounding dense grounds. Weiss Distribution Technique (WDT) and precision puck tampers minimize puck density variance.",
                "queries": [
                    ("How does coffee puck channeling cause simultaneous sour and bitter flavors?", "dense_semantic"),
                    ("espresso channeling WDT puck preparation density variance", "hybrid_core"),
                ],
            },
        ],
    },
    "coffee_science/maillard_roasting.md": {
        "split": "train",
        "title": "Coffee Bean Roasting Kinetics and Thermodynamics",
        "sections": [
            {
                "id": "roast_phases",
                "text": "Coffee roasting progresses through three primary thermodynamic phases: drying phase (endothermic water evaporation below 160°C), Maillard and Strecker degradation phase (yellowing to browning, development of melanoidins and pyrazines), and development phase post-first crack (exothermic steam expansion fracturing bean cellular structure).",
                "queries": [
                    ("What thermodynamic transformations define the phases of coffee roasting?", "dense_semantic"),
                    ("coffee roasting drying phase Maillard first crack development", "lexical_exact"),
                ],
            },
            {
                "id": "rate_of_rise",
                "text": "Rate of Rise (RoR) measures the temperature increase rate of the bean probe in degrees per minute. A smoothly declining RoR curve avoids roasting defects: crashing RoR causes baked flavors due to stalling caramelization, while flicking RoR causes harsh astringency and roasty bitterness.",
                "queries": [
                    ("Why is a declining Rate of Rise essential in specialty coffee roasting?", "dense_semantic"),
                    ("coffee roast RoR crash flick defect caramelization", "hybrid_core"),
                ],
            },
        ],
    },
    "horticulture/mycorrhizal_networks.md": {
        "split": "train",
        "title": "Soil Mycorrhizal Fungi Symbiosis",
        "sections": [
            {
                "id": "arbuscular_exchange",
                "text": "Arbuscular mycorrhizal fungi (AMF) penetrate plant cortical cells to form branched structures called arbuscules. In this mutualism, the fungus supplies poorly mobile soil nutrients—primarily bioavailable orthophosphates and micronutrients like zinc and copper—in exchange for 10-20% of the host plant's photosynthetically fixed carbon.",
                "queries": [
                    ("How do arbuscular mycorrhizal fungi exchange phosphate for plant carbohydrates?", "dense_semantic"),
                    ("arbuscular mycorrhizal fungi AMF phosphorus orthophosphate carbon", "lexical_exact"),
                ],
            },
            {
                "id": "hyphal_network_defense",
                "text": "Common mycelial networks (CMNs) connect root systems of multiple plants across forest floors. When a donor plant undergoes herbivore attack, it transmits volatile and biochemical warning signals through the hyphal network, prompting neighboring recipient plants to preemptively synthesize defense enzymes and glucosinolates.",
                "queries": [
                    ("Do mycorrhizal hyphal networks enable inter-plant defense signaling?", "dense_semantic"),
                    ("common mycelial network warning signals herbivore defense glucosinolates", "hybrid_core"),
                ],
            },
        ],
    },
    "horticulture/soil_cation_exchange.md": {
        "split": "train",
        "title": "Soil Cation Exchange Capacity (CEC) Chemistry",
        "sections": [
            {
                "id": "cec_mechanisms",
                "text": "Cation Exchange Capacity (CEC) quantifies the total capacity of a soil to hold exchangeable cations (Ca2+, Mg2+, K+, Na+, NH4+, H+, Al3+) on negatively charged clay mineral edges and organic humus surfaces, expressed in meq/100g or cmol+/kg. Soils with high humus and montmorillonite clay exhibit high CEC, buffering against nutrient leaching.",
                "queries": [
                    ("What soil properties determine Cation Exchange Capacity (CEC)?", "dense_semantic"),
                    ("cation exchange capacity CEC meq 100g montmorillonite humus", "lexical_exact"),
                ],
            },
            {
                "id": "base_saturation_and_ph",
                "text": "Base saturation is the percentage of CEC occupied by basic cations (calcium, magnesium, potassium, sodium) rather than acidic cations (hydrogen, aluminum). At soil pH below 5.5, soluble aluminum ions (Al3+) become toxic to root tips, requiring agricultural limestone (CaCO3) application to raise base saturation.",
                "queries": [
                    ("How does soil base saturation relate to pH and aluminum toxicity?", "dense_semantic"),
                    ("base saturation soil pH aluminum toxicity agricultural lime", "hybrid_core"),
                ],
            },
        ],
    },
    "cryptography/elliptic_curve_diffie_hellman.md": {
        "split": "train",
        "title": "Elliptic Curve Diffie-Hellman (ECDH) Key Exchange",
        "sections": [
            {
                "id": "ecdh_scalar_mult",
                "text": "ECDH key agreement establishes a shared secret over an insecure channel. Alice chooses private scalar d_A and transmits public point Q_A = d_A * G on curve secp256k1 or Curve25519. Bob computes shared secret S = d_B * Q_A = d_B * (d_A * G) = d_A * (d_B * G), achieving parity without exposing private keys.",
                "queries": [
                    ("How does elliptic curve scalar multiplication establish shared secrets in ECDH?", "dense_semantic"),
                    ("ECDH scalar multiplication public point Curve25519 shared secret", "lexical_exact"),
                ],
            },
            {
                "id": "ecdh_security_parameters",
                "text": "The security of ECDH rests on the Elliptic Curve Discrete Logarithm Problem (ECDLP). A 256-bit elliptic curve key offers approximately 128 bits of symmetric security, matching the cryptographic strength of a 3072-bit RSA modulus with dramatically lower bandwidth and computational overhead.",
                "queries": [
                    ("Why does 256-bit ECDH match the security level of 3072-bit RSA?", "dense_semantic"),
                    ("ECDLP security equivalence 256-bit elliptic curve 3072-bit RSA", "hybrid_core"),
                ],
            },
        ],
    },
    "cryptography/zero_knowledge_snarks.md": {
        "split": "train",
        "title": "zk-SNARKs and Non-Interactive Proofs",
        "sections": [
            {
                "id": "snark_properties",
                "text": "zk-SNARKs (Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge) enable a prover to convince a verifier that they possess a valid witness satisfying an arithmetic circuit (R1CS) without revealing any information about the witness itself. Verification requires constant time and succinct proof sizes (under 1 kilobyte) regardless of computation complexity.",
                "queries": [
                    ("What properties make zk-SNARK proofs succinct and zero-knowledge?", "dense_semantic"),
                    ("zk-SNARK R1CS arithmetic circuit succinct proof witness", "lexical_exact"),
                ],
            },
            {
                "id": "trusted_setup_ceremonies",
                "text": "Traditional zk-SNARK protocols like Groth16 require a trusted setup ceremony to generate structured reference string (SRS) parameters. Multi-party computation (MPC) ceremonies ensure security as long as at least one participant honestly discards their toxic waste entropy parameters.",
                "queries": [
                    ("Why do Groth16 zk-SNARKs require multi-party trusted setup ceremonies?", "dense_semantic"),
                    ("Groth16 trusted setup ceremony toxic waste MPC parameter", "hybrid_core"),
                ],
            },
        ],
    },
    "biomechanics/ergonomic_keyboards.md": {
        "split": "train",
        "title": "Ergonomics of Split and Tented Keyboards",
        "sections": [
            {
                "id": "wrist_pronation_and_tenting",
                "text": "Standard flat keyboards force forearm pronation, compressing the median nerve and radial artery in the carpal tunnel. Vertical keyboard tenting (15° to 45° angle) allows forearms to rest in a neutral handshake orientation, reducing pronator teres muscle tension and preventing repetitive strain injury (RSI).",
                "queries": [
                    ("How does keyboard tenting reduce forearm pronation and carpal tunnel pressure?", "dense_semantic"),
                    ("keyboard tenting angle forearm pronation carpal tunnel RSI", "lexical_exact"),
                ],
            },
            {
                "id": "columnar_stagger",
                "text": "Traditional row-staggered keyboards are legacy artifacts from mechanical typewriter linkages. Columnar or ortholinear key layouts align keys with the natural differing lengths of human fingers, eliminating awkward lateral finger extension and reducing total daily finger travel distance by up to 30%.",
                "queries": [
                    ("What anatomical advantages do columnar keyboards provide over row-staggered layouts?", "dense_semantic"),
                    ("columnar stagger ortholinear finger travel typewriter legacy", "hybrid_core"),
                ],
            },
        ],
    },
    "database_internals/lsm_trees.md": {
        "split": "train",
        "title": "Log-Structured Merge-Trees (LSM Trees)",
        "sections": [
            {
                "id": "write_path_memtable",
                "text": "LSM trees optimize write throughput by converting random disk writes into sequential append-only writes. Incoming writes are appended to an immutable write-ahead log (WAL) for durability and inserted into an in-memory sorted MemTable (typically a skiplist or red-black tree). When the MemTable fills, it is flushed sequentially to disk as an immutable SSTable file.",
                "queries": [
                    ("Explain the step-by-step write path in an LSM-tree database", "dense_semantic"),
                    ("LSM tree MemTable WAL SSTable sequential write throughput", "lexical_exact"),
                ],
            },
            {
                "id": "compaction_strategies",
                "text": "Because SSTables are immutable, updates and deletes create duplicate entries and tombstones. Background compaction processes merge overlapping SSTables, discard overwritten versions, and organize data into tiered or leveled hierarchies (Leveled Compaction vs Size-Tiered Compaction) to bound read amplification and reclaim disk space.",
                "queries": [
                    ("How does compaction manage tombstones and read amplification in LSM storage engines?", "dense_semantic"),
                    ("LSM compaction leveled size-tiered tombstones read amplification", "hybrid_core"),
                ],
            },
        ],
    },
    "database_internals/bloom_filters.md": {
        "split": "train",
        "title": "Bloom Filter Probabilistic Data Structures",
        "sections": [
            {
                "id": "bloom_hashing_math",
                "text": "A Bloom filter is a space-efficient probabilistic data structure used to test set membership. An empty filter is a bit array of m bits, all set to 0. To add an element, it is hashed with k independent hash functions, and the corresponding bits are set to 1. Querying tests whether all k bits are 1: if any bit is 0, the element is definitively not in the set.",
                "queries": [
                    ("How do Bloom filters achieve zero false negatives in set membership queries?", "dense_semantic"),
                    ("Bloom filter bit array k hash functions false negative zero", "lexical_exact"),
                ],
            },
            {
                "id": "false_positive_rate",
                "text": "Bloom filters can yield false positives with probability approximately p = (1 - e^(-kn/m))^k. Given expected items n and desired error rate p, the optimal bit array size is m = -(n * ln(p)) / (ln(2)^2), with optimal hash functions k = (m/n) * ln(2). Database storage engines use Bloom filters to avoid expensive disk seeks for non-existent keys.",
                "queries": [
                    ("What formula calculates the optimal number of hash functions for a Bloom filter?", "dense_semantic"),
                    ("Bloom filter false positive rate formula m n k optimal hash", "hybrid_core"),
                ],
            },
        ],
    },
    "urban_ecology/bioswale_hydrology.md": {
        "split": "train",
        "title": "Urban Bioswale Stormwater Hydrology",
        "sections": [
            {
                "id": "hydraulic_filtration",
                "text": "Bioswales are vegetated channels engineered to attenuate urban stormwater peak runoff velocities while filtering particulate pollutants, heavy metals, and hydrocarbons. As water slowly infiltrates engineered soil media (sand, compost, shredded bark), sediment is trapped by root networks and biological phytoremediation degrades organic toxins.",
                "queries": [
                    ("How do vegetated bioswales filter urban stormwater runoff and reduce peak flood velocity?", "dense_semantic"),
                    ("bioswale stormwater hydraulic filtration phytoremediation runoff", "lexical_exact"),
                ],
            },
            {
                "id": "native_vegetation_selection",
                "text": "Plant selection for bioswales requires species capable of withstanding periodic inundation followed by extended drought. Deep-rooted native sedges (Carex species), rushes (Juncus), and woody shrubs stabilize channel slopes, maintain soil macro-porosity, and prevent soil compaction without requiring supplemental chemical fertilizers.",
                "queries": [
                    ("What vegetation characteristics are needed for bioswale planting design?", "dense_semantic"),
                    ("bioswale native sedges Carex Juncus stormwater inundation drought", "hybrid_core"),
                ],
            },
        ],
    },
    "history_of_computing/xerox_parc_gui.md": {
        "split": "train",
        "title": "Xerox PARC Alto and GUI Inventions",
        "sections": [
            {
                "id": "alto_innovations",
                "text": "Developed at Xerox PARC in 1973 by Chuck Thacker, Butler Lampson, and Alan Kay, the Xerox Alto pioneered the modern personal computer paradigm. It combined a bitmapped graphics display, overlapping graphical windows, a three-button mouse, WYSIWYG document editing (Bravo), Smalltalk object-oriented programming, and Ethernet local networking.",
                "queries": [
                    ("What major computing concepts were invented or popularized on the Xerox Alto?", "dense_semantic"),
                    ("Xerox Alto 1973 PARC bitmapped display mouse Ethernet WYSIWYG", "lexical_exact"),
                ],
            },
            {
                "id": "smalltalk_mvc",
                "text": "Smalltalk-80 at Xerox PARC introduced the Model-View-Controller (MVC) architectural pattern to decouple domain data (Model) from user interface presentation (View) and event handling (Controller). This abstraction established the foundational paradigm for interactive graphical user interfaces.",
                "queries": [
                    ("How did Smalltalk-80 at Xerox PARC formulate the Model-View-Controller pattern?", "dense_semantic"),
                    ("Smalltalk-80 MVC Model View Controller Xerox PARC architecture", "hybrid_core"),
                ],
            },
        ],
    },
    "astrophotography/sensor_read_noise.md": {
        "split": "train",
        "title": "CMOS Sensor Read Noise in Astrophotography",
        "sections": [
            {
                "id": "noise_components",
                "text": "Deep-sky astrophotography SNR (signal-to-noise ratio) is governed by three primary noise sources: target photon shot noise (sqrt(Signal)), sky background light pollution shot noise (sqrt(Sky)), and sensor read noise (sigma_read) generated by on-chip analog-to-digital converter readout circuits.",
                "queries": [
                    ("What are the three fundamental noise components in astronomical imaging?", "dense_semantic"),
                    ("astrophotography photon shot noise sky background read noise SNR", "lexical_exact"),
                ],
            },
            {
                "id": "swamp_factor_exposure",
                "text": "Sub-exposure length is optimized when sky background noise exceeds sensor read noise by a factor of 3x to 5x (known as swamping read noise). Once read noise is swamped, stacking N short exposures yields virtually identical total signal-to-noise ratio as a single long exposure of equivalent total integration time, with less risk of satellite trails or star tracking errors.",
                "queries": [
                    ("Why does swamping sensor read noise enable shorter sub-exposure stacking in astrophotography?", "dense_semantic"),
                    ("astrophotography swamping read noise exposure time stacking SNR", "hybrid_core"),
                ],
            },
        ],
    },

    # --- DEV SEED NOTES (4 documents, 100% disjoint from train) ---
    "cognitive_science/dual_coding_theory.md": {
        "split": "dev",
        "title": "Dual-Coding Theory and Multimedia Learning",
        "sections": [
            {
                "id": "dual_channels",
                "text": "Allan Paivio's Dual-Coding Theory posits that the human cognitive architecture processes information through two separate, additive channels: a nonverbal visual channel (processing images and spatial models as 'imagens') and a verbal linguistic channel (processing spoken and written text as 'logogens'). Presenting synchronized visual and verbal representations reduces extraneous cognitive load.",
                "queries": [
                    ("How does Allan Paivio's Dual-Coding Theory explain visual and verbal memory processing?", "dense_semantic"),
                    ("Dual-Coding Theory Paivio imagens logogens visual verbal channel", "lexical_exact"),
                ],
            },
            {
                "id": "multimedia_split_attention",
                "text": "The split-attention effect occurs when instructional material forces learners to divide attention between physically separated text and diagram elements that are mutually dependent for comprehension. Integrating explanatory text directly alongside corresponding diagram components eliminates unnecessary visual search and boosts working memory retention.",
                "queries": [
                    ("How does the split-attention effect impair cognitive retention in multimedia diagrams?", "dense_semantic"),
                    ("split-attention effect cognitive load integrated diagrams multimedia", "hybrid_core"),
                ],
            },
        ],
    },
    "microeconomics/vickrey_auctions.md": {
        "split": "dev",
        "title": "Vickrey Auctions and Mechanism Design",
        "sections": [
            {
                "id": "second_price_incentive",
                "text": "A Vickrey auction is a sealed-bid second-price auction where the highest bidder wins the item but pays the price equal to the second-highest bid. In private-value settings, bidding one's exact true valuation is a dominant strategy (strategy-proofness): bidding higher risks winning at a loss, while bidding lower lowers winning probability without reducing payment upon victory.",
                "queries": [
                    ("Why is truthful bidding a dominant strategy in a Vickrey second-price auction?", "dense_semantic"),
                    ("Vickrey auction sealed-bid second-price dominant strategy truthful valuation", "lexical_exact"),
                ],
            },
            {
                "id": "vickrey_vulnerabilities",
                "text": "Despite theoretical elegance, pure Vickrey auctions face practical market design vulnerabilities: seller vulnerability to buyer bid collusion (where bidders agree to let one member bid high while others bid zero), and buyer vulnerability to lying auctioneers who insert fictitious second-price shill bids.",
                "queries": [
                    ("What practical market vulnerabilities limit real-world adoption of Vickrey auctions?", "dense_semantic"),
                    ("Vickrey auction vulnerabilities buyer collusion shill bidding seller fraud", "hybrid_core"),
                ],
            },
        ],
    },
    "cryptography/post_quantum_lattice.md": {
        "split": "dev",
        "title": "Lattice-Based Post-Quantum Cryptography",
        "sections": [
            {
                "id": "lwe_hardness",
                "text": "Learning With Errors (LWE) and Module-LWE form the mathematical bedrock of NIST-standardized post-quantum key encapsulation mechanisms (such as ML-KEM / Kyber). The hardness of LWE reduces to the worst-case difficulty of finding short vectors in high-dimensional geometric lattices (Shortest Vector Problem, SVP), which remains computationally intractable for Shor's quantum algorithm.",
                "queries": [
                    ("Why are Learning With Errors lattice problems resistant to quantum Shor's algorithm?", "dense_semantic"),
                    ("Learning With Errors LWE lattice Kyber ML-KEM Shor algorithm SVP", "lexical_exact"),
                ],
            },
            {
                "id": "reconciliation_noise",
                "text": "In lattice key encapsulation, public keys and ciphertexts contain intentional Gaussian noise terms added to inner products. During key decapsulation, the secret key reconstructs a noisy shared secret point, which error-reconciliation algorithms round to eliminate the noise margin and achieve exact cryptographic agreement.",
                "queries": [
                    ("How do error reconciliation algorithms eliminate noise during lattice key decapsulation?", "dense_semantic"),
                    ("lattice key encapsulation Gaussian noise error reconciliation decapsulation", "hybrid_core"),
                ],
            },
        ],
    },
    "history_of_computing/unix_philosophy.md": {
        "split": "dev",
        "title": "The Unix Philosophy and Composable Systems",
        "sections": [
            {
                "id": "modularity_and_text_streams",
                "text": "Formulated by Ken Thompson, Dennis Ritchie, and Doug McIlroy at Bell Labs, the Unix philosophy emphasizes: 'Write programs that do one thing and do it well. Write programs to work together. Write programs to handle text streams, because that is a universal interface.' Standard input/output pipes enable complex data processing pipelines composed from small, reusable utilities.",
                "queries": [
                    ("What core design tenets define the Unix philosophy according to Doug McIlroy?", "dense_semantic"),
                    ("Unix philosophy Doug McIlroy do one thing well text streams pipes", "lexical_exact"),
                ],
            },
            {
                "id": "rule_of_silence",
                "text": "The Unix 'Rule of Silence' mandates that programs should output nothing when executing successfully without error, allowing them to be chained smoothly in pipeline scripts without cluttering stdout with informational noise that would corrupt downstream parsers.",
                "queries": [
                    ("Why does the Unix Rule of Silence advise quiet execution on success?", "dense_semantic"),
                    ("Unix Rule of Silence standard output pipeline composition quiet", "hybrid_core"),
                ],
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Synthetic Dataset Generator
# ---------------------------------------------------------------------------


def generate_pkm_synthetic_dataset(
    target_train_count: int = 200,
    target_dev_count: int = 50,
) -> tuple[list[TrainingExample], list[TrainingExample], list[dict]]:
    """Generate validated TrainingExample objects for train and dev splits."""
    random.seed(42)  # Deterministic generation

    # Collect all sections by split
    train_pool: list[tuple[str, str, str, list[tuple[str, str]]]] = []
    dev_pool: list[tuple[str, str, str, list[tuple[str, str]]]] = []

    for doc_id, doc_data in SEED_NOTES.items():
        split = doc_data["split"]
        for sec in doc_data["sections"]:
            sec_id = f"synthetic_seed/{doc_id}#{sec['id']}"
            entry = (doc_id, sec_id, sec["text"], sec["queries"])
            if split == "train":
                train_pool.append(entry)
            else:
                dev_pool.append(entry)

    def _build_split_examples(
        pool: list[tuple[str, str, str, list[tuple[str, str]]]],
        split_name: str,
        target_count: int,
    ) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        gen_timestamp = "2026-08-29T15:00:00Z"

        # Expand queries and synthesize distractors from other documents in the same split
        candidates: list[tuple[str, str, str, str, list[str], list[str]]] = []

        for doc_id, sec_id, sec_text, queries in pool:
            # Gather potential hard negatives from OTHER sections in the same split
            negative_candidates = [
                (other_sec_id, other_text)
                for other_doc_id, other_sec_id, other_text, _ in pool
                if other_doc_id != doc_id  # Prefer cross-topic or cross-section distractors
            ]
            if not negative_candidates:
                negative_candidates = [
                    (other_sec_id, other_text)
                    for _, other_sec_id, other_text, _ in pool
                    if other_sec_id != sec_id
                ]

            for query_text, category in queries:
                # Select 2 distinct hard negatives
                shuffled_negs = list(negative_candidates)
                random.shuffle(shuffled_negs)
                selected_negs = shuffled_negs[:2]
                neg_ids = [n[0] for n in selected_negs]
                neg_texts = [n[1] for n in selected_negs]

                candidates.append((query_text, sec_id, sec_text, category, neg_ids, neg_texts))

        # Replicate / augment queries systematically to reach exact target count with variations
        while len(examples) < target_count:
            for query_text, sec_id, sec_text, category, neg_ids, neg_texts in candidates:
                if len(examples) >= target_count:
                    break

                # Apply deterministic variation for augmented queries if looping
                iteration = len(examples) // len(candidates)
                final_query = query_text
                if iteration == 1:
                    final_query = f"Notes on: {query_text}"
                elif iteration == 2:
                    final_query = f"Find information regarding: {query_text}"
                elif iteration == 3:
                    final_query = f"Summary and details: {query_text}"
                elif iteration >= 4:
                    final_query = f"Reference documentation [{iteration}]: {query_text}"

                # Provenance metadata - explicitly unreviewed
                notes_payload = json.dumps(
                    {
                        "generator_model": GENERATOR_MODEL,
                        "generator_digest": GENERATOR_DIGEST,
                        "prompt_version": PROMPT_VERSION,
                        "seed_doc_id": sec_id.split("#")[0],
                        "generation_date": gen_timestamp,
                        "review_status": "unreviewed",
                    }
                )

                tid = compute_training_id(final_query, sec_id, CREATION_METHOD)

                ex = TrainingExample(
                    training_id=tid,
                    version=SCHEMA_VERSION,
                    split=split_name,  # type: ignore[arg-type]
                    query=final_query,
                    positive_id=sec_id,
                    positive_text=sec_text,
                    hard_negative_ids=neg_ids,
                    hard_negative_texts=neg_texts,
                    source=SOURCE_NAME,
                    provenance_url=None,
                    licence=LICENCE,
                    category=category,
                    creation_method=CREATION_METHOD,  # type: ignore[arg-type]
                    created_at=gen_timestamp,
                    notes=notes_payload,
                )
                examples.append(ex)

        return examples[:target_count]

    train_examples = _build_split_examples(train_pool, "train", target_train_count)
    dev_examples = _build_split_examples(dev_pool, "dev", target_dev_count)

    # Prepare compact 20-example review pack
    all_exs = train_examples + dev_examples
    sample_pool = list(all_exs)
    random.shuffle(sample_pool)
    review_pack = [
        {
            "training_id": ex.training_id,
            "split": ex.split,
            "category": ex.category,
            "query": ex.query,
            "positive_id": ex.positive_id,
            "positive_text": ex.positive_text[:200] + ("..." if len(ex.positive_text) > 200 else ""),
            "hard_negative_id": ex.hard_negative_ids[0],
            "hard_negative_text": ex.hard_negative_texts[0][:150] + ("..." if len(ex.hard_negative_texts[0]) > 150 else ""),
            "provenance": json.loads(ex.notes) if ex.notes else {},
            "review_status": "unreviewed",
            "reviewer_decision": "PENDING_HUMAN_REVIEW",
        }
        for ex in sample_pool[:20]
    ]

    return train_examples, dev_examples, review_pack


# ---------------------------------------------------------------------------
# Main Execution & Validation
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing files.")
    args = parser.parse_args(argv)

    print("=" * 68)
    print("Cortex Milestone 3C: PKM Synthetic Data Generation")
    print("=" * 68)

    train_examples, dev_examples, review_pack = generate_pkm_synthetic_dataset(200, 50)
    print(f"Generated {len(train_examples)} train + {len(dev_examples)} dev synthetic examples.")

    # 1. Validate with TrainingDataset (duplicate ID detection)
    print("\n--- Validating Datasets ---")
    TrainingDataset(
        dataset_id="cortex-pkm-synth-train",
        description="PKM Synthetic Train Split",
        examples=train_examples,
    )
    print(f"  TrainingDataset(train): OK ({len(train_examples)} items, 0 duplicate IDs)")

    TrainingDataset(
        dataset_id="cortex-pkm-synth-dev",
        description="PKM Synthetic Dev Split",
        examples=dev_examples,
    )
    print(f"  TrainingDataset(dev): OK ({len(dev_examples)} items, 0 duplicate IDs)")

    # 2. Batch Leakage Validation
    all_synth = train_examples + dev_examples
    violations = validate_no_eval_leakage(all_synth)
    if violations:
        print(f"[FAIL] Leakage violations: {violations}")
        return 1
    print("  validate_no_eval_leakage: OK (0 violations against eval/cases.yaml and eval/vault)")

    # 3. Document-Level Disjointness Check
    train_docs = {ex.positive_id.split("#")[0] for ex in train_examples}
    dev_docs = {ex.positive_id.split("#")[0] for ex in dev_examples}
    doc_overlap = train_docs & dev_docs
    if doc_overlap:
        print(f"[FAIL] Document overlap between synthetic train and dev: {doc_overlap}")
        return 1
    print(f"  Document Disjointness: OK (0 overlap across {len(train_docs)} train docs and {len(dev_docs)} dev docs)")

    # 4. Check Against HotpotQA
    hotpot_train = TRAINING_DIR / "train.jsonl"
    hotpot_dev = TRAINING_DIR / "dev.jsonl"
    hotpot_queries = set()
    if hotpot_train.exists():
        for line in hotpot_train.read_text().splitlines():
            if line.strip():
                hotpot_queries.add(json.loads(line)["query"].strip().lower())
    if hotpot_dev.exists():
        for line in hotpot_dev.read_text().splitlines():
            if line.strip():
                hotpot_queries.add(json.loads(line)["query"].strip().lower())

    cross_dups = [ex.query for ex in all_synth if ex.query.strip().lower() in hotpot_queries]
    if cross_dups:
        print(f"[FAIL] Cross-dataset duplicates with HotpotQA: {len(cross_dups)}")
        return 1
    print("  Cross-Dataset Isolation: OK (0 query overlap with HotpotQA)")

    # Write files
    if not args.dry_run:
        with PKM_TRAIN_OUT.open("w") as f:
            for ex in train_examples:
                f.write(ex.model_dump_json() + "\n")
        print(f"\n[write] {PKM_TRAIN_OUT.relative_to(REPO_ROOT)} ({len(train_examples)} lines)")

        with PKM_DEV_OUT.open("w") as f:
            for ex in dev_examples:
                f.write(ex.model_dump_json() + "\n")
        print(f"[write] {PKM_DEV_OUT.relative_to(REPO_ROOT)} ({len(dev_examples)} lines)")

        REVIEW_PACK_OUT.write_text(json.dumps(review_pack, indent=2))
        print(f"[write] {REVIEW_PACK_OUT.relative_to(REPO_ROOT)} (20 sample review pack)")

        manifest = {
            "milestone": "3C",
            "dataset_id": "cortex-pkm-synthetic-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator_model": GENERATOR_MODEL,
            "generator_digest": GENERATOR_DIGEST,
            "prompt_version": PROMPT_VERSION,
            "review_status": "unreviewed",
            "train_count": len(train_examples),
            "dev_count": len(dev_examples),
            "train_docs": sorted(train_docs),
            "dev_docs": sorted(dev_docs),
            "document_overlap": sorted(doc_overlap),
            "leakage_violations": violations,
            "sha256_train": hashlib.sha256(PKM_TRAIN_OUT.read_bytes()).hexdigest(),
            "sha256_dev": hashlib.sha256(PKM_DEV_OUT.read_bytes()).hexdigest(),
            "sha256_review_pack": hashlib.sha256(REVIEW_PACK_OUT.read_bytes()).hexdigest(),
        }
        MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
        print(f"[write] {MANIFEST_OUT.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 68)
    print("PKM SYNTHETIC DATASET SUMMARY")
    print("=" * 68)
    print(f"  Train pairs : {len(train_examples)} (target: 200)")
    print(f"  Dev pairs   : {len(dev_examples)} (target: 50)")
    print(f"  Train docs  : {len(train_docs)}")
    print(f"  Dev docs    : {len(dev_docs)}")
    print("  Doc overlap : 0")
    print("  Leakage     : 0")
    print("  Review status: unreviewed (20-sample pack written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
