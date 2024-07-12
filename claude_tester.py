import os
import subprocess
import sys
import venv
import json
import time
import psutil
import coverage
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import logging

class ClaudeTester:
    def __init__(self, venv_name: str, python_version: str = "3.8"):
        self.venv_name = venv_name
        self.venv_path = os.path.abspath(venv_name)
        self.python_version = python_version
        self.log_file = f"{venv_name}_log.txt"
        self.setup_logging()
        self.start_time = time.time()
        self.env_vars: Dict[str, str] = {}
        self.installed_packages: Dict[str, str] = {}

    def setup_logging(self):
        logging.basicConfig(filename=self.log_file, level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def log(self, message: str, level: str = "info"):
        getattr(self.logger, level)(message)
        print(f"[{level.upper()}] {message}")

    def create_venv(self):
        self.log(f"Creating virtual environment '{self.venv_name}' with Python {self.python_version}")
        try:
            subprocess.run([f"python{self.python_version}", "-m", "venv", self.venv_path], check=True, capture_output=True)
            self.log("Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            self.log(f"Error creating virtual environment: {e.stderr.decode()}", "error")
            raise

    def set_env_var(self, key: str, value: str):
        self.env_vars[key] = value
        self.log(f"Set environment variable: {key}={value}")

    def install_dependencies(self, dependencies: List[str]):
        self.log(f"Installing dependencies: {', '.join(dependencies)}")
        pip_path = os.path.join(self.venv_path, 'Scripts', 'pip') if os.name == 'nt' else os.path.join(self.venv_path, 'bin', 'pip')
        for dep in dependencies:
            try:
                output = subprocess.run([pip_path, 'install', dep], check=True, capture_output=True, text=True)
                version = subprocess.run([pip_path, 'show', dep.split('==')[0]], capture_output=True, text=True).stdout
                version = next((line.split(': ')[1] for line in version.split('\n') if line.startswith('Version: ')), 'Unknown')
                self.installed_packages[dep.split('==')[0]] = version
                self.log(f"Successfully installed {dep} (version: {version})")
            except subprocess.CalledProcessError as e:
                self.log(f"Error installing {dep}: {e.stderr}", "error")
                raise

    async def run_code_with_timeout(self, code: str, timeout: int = 30, max_memory: int = 500):
        self.log("Running code in virtual environment")
        python_path = os.path.join(self.venv_path, 'Scripts', 'python') if os.name == 'nt' else os.path.join(self.venv_path, 'bin', 'python')
        with open('temp_script.py', 'w') as f:
            f.write(code)

        process = await asyncio.create_subprocess_exec(
            python_path, 'temp_script.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.env_vars}
        )

        try:
            start_time = time.time()
            while process.returncode is None:
                if time.time() - start_time > timeout:
                    process.terminate()
                    raise TimeoutError(f"Code execution timed out after {timeout} seconds")
                
                mem_info = psutil.virtual_memory()
                if mem_info.percent > max_memory:
                    process.terminate()
                    raise MemoryError(f"Memory usage exceeded {max_memory}% of total memory")
                
                await asyncio.sleep(0.1)

            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"Code execution failed with return code {process.returncode}: {stderr.decode()}")

            self.log("Code execution successful")
            return stdout.decode()
        finally:
            os.remove('temp_script.py')

    def run_tests_with_coverage(self, test_code: str):
        self.log("Running tests with coverage")
        cov = coverage.Coverage()
        cov.start()

        with open('temp_test.py', 'w') as f:
            f.write(test_code)

        python_path = os.path.join(self.venv_path, 'Scripts', 'python') if os.name == 'nt' else os.path.join(self.venv_path, 'bin', 'python')
        try:
            result = subprocess.run([python_path, '-m', 'unittest', 'temp_test.py'], capture_output=True, text=True, check=True)
            self.log("Tests completed successfully")
        except subprocess.CalledProcessError as e:
            self.log(f"Tests failed: {e.stderr}", "error")
            raise
        finally:
            os.remove('temp_test.py')
            cov.stop()
            cov.save()

        coverage_report = cov.report(show_missing=True)
        self.log(f"Coverage report:\n{coverage_report}")
        return result.stdout, coverage_report

    def generate_report(self) -> Dict[str, Any]:
        end_time = time.time()
        execution_time = end_time - self.start_time

        report = {
            "venv_name": self.venv_name,
            "python_version": self.python_version,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "installed_packages": self.installed_packages,
            "environment_variables": self.env_vars,
            "log_file": self.log_file
        }

        report_path = f"{self.venv_name}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        self.log(f"Report generated: {report_path}")
        return report

async def claude_tester(venv_name: str, dependencies: List[str], code: str, test_code: str = None, python_version: str = "3.8", env_vars: Dict[str, str] = None, timeout: int = 30, max_memory: int = 500):
    tester = ClaudeTester(venv_name, python_version)
    try:
        tester.create_venv()
        tester.install_dependencies(dependencies)
        
        if env_vars:
            for key, value in env_vars.items():
                tester.set_env_var(key, value)
        
        output = await tester.run_code_with_timeout(code, timeout, max_memory)
        
        test_results = None
        if test_code:
            test_output, coverage_report = tester.run_tests_with_coverage(test_code)
            test_results = {"output": test_output, "coverage": coverage_report}
        
        report = tester.generate_report()
        return {
            "execution_output": output,
            "test_results": test_results,
            "report": report
        }
    except Exception as e:
        tester.log(f"Error in claude_tester: {str(e)}", "error")
        report = tester.generate_report()
        return {
            "error": str(e),
            "report": report
        }

# Example usage (commented out):
# if __name__ == "__main__":
#     venv_name = "complex_test_env"
#     dependencies = ["numpy==1.21.0", "pandas==1.3.0"]
#     code = """
#     import numpy as np
#     import pandas as pd
#     print("NumPy version:", np.__version__)
#     print("Pandas version:", pd.__version__)
#     df = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])
#     print(df.describe())
#     """
#     test_code = """
#     import unittest
#     import numpy as np
#     import pandas as pd
#
#     class TestDataFrame(unittest.TestCase):
#         def test_df_shape(self):
#             df = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])
#             self.assertEqual(df.shape, (5, 3))
#
#         def test_df_columns(self):
#             df = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])
#             self.assertEqual(list(df.columns), ['A', 'B', 'C'])
#     """
#     env_vars = {"PYTHONPATH": "/custom/path"}
#     
#     import asyncio
#     result = asyncio.run(claude_tester(venv_name, dependencies, code, test_code, env_vars=env_vars))
#     print(json.dumps(result, indent=2))
