

"""Library to run Mmseqs from Python."""

import os
import tempfile
import subprocess
import time
from typing import List, Dict, Optional


from absl import logging
from onescience.flax_models.alphafold3.data import parsers
from onescience.flax_models.alphafold3.data.tools import msa_tool
from onescience.flax_models.alphafold3.data.tools import subprocess_utils
import shlex

import shutil
import re


class Mmseqs(msa_tool.MsaTool):
    
    _database_types = ["mgnify", "uniprot_cluster_annot", "uniref90", "small_bfd"]

    def __init__(
        self,
        *,
        binary_path: str,
        database_path: str,
        n_cpu: int = 8,
        e_value: float = 1e-4,
        max_sequences: int = 5000,
        use_gpu: int = 1,
        msa_format_mode: int = 4,
        mmseqs_options: str = "",
        result2msa_options: str = "",
    ):
        self.binary_path = binary_path
        self.database_path = database_path
        
        subprocess_utils.check_binary_exists(path=self.binary_path, name='MMseqs')
        
        if not os.path.exists(self.database_path):
            raise ValueError(f'Database not found: {database_path}')
        
        self.n_cpu = n_cpu
        self.e_value = e_value
        self.max_sequences = max_sequences
        self.use_gpu = use_gpu
        self.msa_format_mode = msa_format_mode
        self.mmseqs_options = mmseqs_options
        self.result2msa_options = result2msa_options

    def _get_gpu_memory_gb(self) -> float:
        hip_devices = os.environ.get('HIP_VISIBLE_DEVICES', '')
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')

        hy_smi = shutil.which('hy-smi')
        nvidia_smi = shutil.which('nvidia-smi')

        prefer_hip = bool(hip_devices or hy_smi)

        device_id: Optional[str] = None
        if prefer_hip and hip_devices:
            ids = [d.strip() for d in hip_devices.split(',') if d.strip()]
            device_id = ids[0] if ids else None
        elif cuda_devices:
            ids = [d.strip() for d in cuda_devices.split(',') if d.strip()]
            device_id = ids[0] if ids else None

        # 1) hy-smi (HIP path)
        if hy_smi:
            try:
                mem = self._get_gpu_memory_gb_via_hy_smi(hy_smi, device_id)
                if mem is not None:
                    logging.info(f"GPU memory (hy-smi): {mem:.1f}GB")
                    return mem
            except Exception as e:  # noqa: BLE001
                logging.warning(f"hy-smi query failed: {e}")

        # 2) NVIDIA fallback when CUDA devices are provided
        if nvidia_smi and cuda_devices:
            try:
                ids = [d.strip() for d in cuda_devices.split(',') if d.strip()]
                if ids:
                    cmd = [
                        nvidia_smi,
                        '--query-gpu=memory.total',
                        '--format=csv,noheader,nounits',
                        f'--id={ids[0]}',
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    memory_mb = float(result.stdout.strip())
                    return memory_mb / 1024.0
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
                logging.warning(f"nvidia-smi query failed: {e}")

        logging.info("Falling back to default GPU memory: 40GB")
        return 40.0

    def _parse_memory_gb_from_text(self, text: str) -> Optional[float]:
        units = {
            'gib': 1.0,
            'gb': 1.0,
            'mib': 1.0 / 1024.0,
            'mb': 1.0 / 1024.0,
            'kib': 1.0 / (1024.0 * 1024.0),
            'kb': 1.0 / (1024.0 * 1024.0),
            'b': 1.0 / (1024.0 * 1024.0 * 1024.0),
        }
        pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB|KiB|KB|B)", re.IGNORECASE)
        candidates = []
        for match in pattern.finditer(text):
            value = float(match.group(1))
            unit = match.group(2).lower()
            factor = units.get(unit, None)
            if factor is not None:
                candidates.append(value * factor)
        if candidates:
            return max(candidates)
        # rocm-smi often prints bytes as: Total Memory (B): 17163091968
        bytes_pattern = re.compile(r"Total\s*Memory\s*\(B\)\s*[:=]\s*(\d{6,})", re.IGNORECASE)
        m = bytes_pattern.search(text)
        if m:
            try:
                return float(m.group(1)) / (1024.0 * 1024.0 * 1024.0)
            except ValueError:
                return None
        # generic Total: <bytes>
        generic_bytes = re.compile(r"Total\s*[:=]\s*(\d{6,})", re.IGNORECASE)
        m2 = generic_bytes.search(text)
        if m2:
            try:
                return float(m2.group(1)) / (1024.0 * 1024.0 * 1024.0)
            except ValueError:
                return None
        return None

    def _get_gpu_memory_gb_via_hy_smi(self, hy_smi_path: str, device_id: Optional[str]) -> Optional[float]:
        candidate_cmds = []
        # 1) Plain text output exactly as user provided; most reliable for MiB lines
        if device_id is not None:
            candidate_cmds.append([hy_smi_path, '--showmeminfo', 'vram', '-d', str(device_id)])
        candidate_cmds.append([hy_smi_path, '--showmeminfo', 'vram'])
        # 2) CSV/JSON (may not include units next to values, kept as fallback)
        if device_id is not None:
            candidate_cmds.append([hy_smi_path, '--showmeminfo', 'vram', '--csv', '-d', str(device_id)])
        candidate_cmds.append([hy_smi_path, '--showmeminfo', 'vram', '--csv'])
        if device_id is not None:
            candidate_cmds.append([hy_smi_path, '--showmeminfo', 'vram', '--json', '-d', str(device_id)])
        candidate_cmds.append([hy_smi_path, '--showmeminfo', 'vram', '--json'])
        # 3) All-in-one info as a last resort
        candidate_cmds.append([hy_smi_path, '-a'])

        for cmd in candidate_cmds:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Prefer exact vram parser for hy-smi
                parsed = self._parse_hysmi_vram_total_gb(result.stdout, device_id)
                if parsed is None:
                    parsed = self._parse_memory_gb_from_text(result.stdout)
                if parsed is not None and parsed > 0:
                    return parsed
            except subprocess.CalledProcessError:
                continue
        return None

    def _parse_hysmi_vram_total_gb(self, text: str, device_id: Optional[str]) -> Optional[float]:
        # Match lines like: HCU[0]          : vram Total Memory (MiB): 65520
        pattern = re.compile(r"HCU\[(\d+)\].*?vram\s+Total\s+Memory\s*\(MiB\)\s*:\s*(\d+)", re.IGNORECASE)
        matches = pattern.findall(text)
        if not matches:
            return None
        values_gb = []
        for dev, mib_str in matches:
            try:
                if device_id is not None and str(device_id) != str(dev):
                    continue
                mib = float(mib_str)
                values_gb.append(mib / 1024.0)
            except ValueError:
                continue
        if not values_gb:
            return None
        return max(values_gb)

    
    
    def _get_database_type(self, database_path: str) -> str:
        path_lower = database_path.lower()
        for db_type in self._database_types:
            if db_type in path_lower:
                return db_type
        return "unknown"
    
    def _calculate_memory_allocation(self, database_path: str) -> int:
        db_type = self._get_database_type(database_path)
        if db_type == "unknown":
            return 8
        
        if not self.use_gpu:
            return 8
        
        gpu_memory_gb = self._get_gpu_memory_gb()
        
        available_memory = max(0, gpu_memory_gb)
        
        if db_type == "mgnify":
            small_dbs_memory = 3 * 2
            allocated_memory = int(available_memory - small_dbs_memory)
            allocated_memory = max(2, allocated_memory)
        else:
            allocated_memory = 2
        
        logging.info(f"Database {db_type}: allocated {allocated_memory}GB GPU memory")
        return allocated_memory

    def query(self, target_sequence: str) -> msa_tool.MsaToolResult:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_fasta = os.path.join(tmp_dir, 'query.fasta')
            subprocess_utils.create_query_fasta_file(target_sequence, input_fasta)
            
            query_db = os.path.join(tmp_dir, 'queryDB')
            self._run_createdb(input_fasta, query_db)
            
            result_db = os.path.join(tmp_dir, 'resultDB')
            self._run_search(query_db, result_db)  
            
            output_sto = os.path.join(tmp_dir, 'output.sto')
            self._run_result2msa(query_db, result_db, output_sto)
            
            with open(output_sto) as f:
                a3m = self._parse_output(f)
        
        return msa_tool.MsaToolResult(
            target_sequence=target_sequence,
            a3m=a3m,
            e_value=self.e_value
        )

    def _run_search(self, input_db: str, output_db: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                self.binary_path,
                'search',
                input_db,
                self.database_path,
                output_db,
                tmp_dir,
                '--threads', str(self.n_cpu),
                '-e', str(self.e_value),
                '--gpu', str(self.use_gpu),
                *shlex.split(self.mmseqs_options)
            ]
            
            if self.use_gpu:
                max_gpu_mem = self._calculate_memory_allocation(self.database_path)
                cmd.extend(['--max-gpu-mem', f'{max_gpu_mem}G'])
                logging.info(f"Running MMseqs search with max GPU memory: {max_gpu_mem}GB")
            else:
                logging.info("Running MMseqs search in CPU mode")
            
            subprocess_utils.run(
                cmd=cmd,
                cmd_name='MMseqs2 search',
                log_stderr=True
            )

    def _run_createdb(self, input_fasta: str, output_db: str):
        cmd = [self.binary_path, 'createdb', input_fasta, output_db]
        subprocess_utils.run(cmd, 'MMseqs2 createdb')

    def _run_result2msa(self, query_db: str, result_db: str, output_sto: str):
        cmd = [
            self.binary_path,
            'result2msa',
            query_db,
            self.database_path,
            result_db,
            output_sto,
            '--msa-format-mode', str(self.msa_format_mode),
            *shlex.split(self.result2msa_options)
        ]
        subprocess_utils.run(cmd, 'MMseqs2 result2msa')

    def _parse_output(self, file_handle):
        try:
            return parsers.convert_stockholm_to_a3m(file_handle, self.max_sequences)
        except Exception as e:
            logging.warning(f"Stockholm parse failed: {e}")
            
            file_handle.seek(0)
            content = file_handle.read()
            debug_file = f"/tmp/stockholm_debug_{int(time.time())}.sto"
            with open(debug_file, 'w') as f:
                f.write(content)
            logging.info(f"Saved debug file: {debug_file}")
            
            file_handle.seek(0)
            return parsers.convert_mmseqs_stockholm_to_a3m(file_handle, self.max_sequences)


