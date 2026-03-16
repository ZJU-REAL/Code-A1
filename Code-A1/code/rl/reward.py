import json
import subprocess
import tempfile
import os
import sys
import traceback
import threading
import queue
from typing import Dict, List, Tuple, Any
import torch
import numpy as np
import re
from collections import defaultdict
import string
import time

from verl import DataProto
from verl.workers.reward_manager import register

from sandbox_fusion import run_code, run_concurrent, RunCodeRequest, RunStatus, CommandRunStatus, RunCodeResponse
from sandbox_fusion.models import CommandRunResult

import ast
import textwrap

def extract_imps_fns(code_list: List[str]):
	cleaned_codes = []
	global_imports = set()
	global_func_names = set()

	class CodeProcessor(ast.NodeTransformer):
		def visit_Import(self, node):
			global_imports.add(ast.unparse(node))
			return None
		def visit_ImportFrom(self, node):
			global_imports.add(ast.unparse(node))
			return None

		def visit_FunctionDef(self, node):
			global_func_names.add(node.name)
			return node

		def visit_AsyncFunctionDef(self, node):
			global_func_names.add(node.name)
			return node

	for code in code_list:
		try:
			tree = ast.parse(code)
			new_tree = CodeProcessor().visit(tree)
			ast.fix_missing_locations(new_tree)
			cleaned_codes.append(ast.unparse(new_tree))
		except Exception:
			cleaned_codes.append(code)

	return cleaned_codes, global_imports, global_func_names

def extract_calls(script_content):
	extracted_calls = []

	clean_pattern = re.compile(r'("[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'|#.*)')

	lines = script_content.split('\n')
	buffer = []
	balance = 0

	for line in lines:
		stripped_line = line.strip()

		if not stripped_line:
			continue

		if balance < 0:
			buffer = []
			balance = 0

		buffer.append(line)

		clean_line = clean_pattern.sub('', stripped_line)

		balance += clean_line.count('(') + clean_line.count('[') + clean_line.count('{')
		balance -= clean_line.count(')') + clean_line.count(']') + clean_line.count('}')

		if balance == 0 and not stripped_line.endswith('\\'):
			block_code = "\n".join(buffer)
			buffer = []

			try:
				tree = ast.parse(block_code.strip())

				for node in tree.body:
					if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
						call_source = ast.unparse(node.value)
						extracted_calls.append(call_source)

			except Exception:
				pass

	return extracted_calls

def extract_asserts(script_content):
	extracted_cases = []
	env = {}

	clean_pattern = re.compile(r'("[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'|#.*)')

	lines = script_content.split('\n')
	buffer = []
	balance = 0

	for line in lines:
		stripped_line = line.strip()
		if not stripped_line:
			continue

		if balance < 0 or (balance > 0 and stripped_line.startswith("assert ")):
			buffer = []
			balance = 0

		buffer.append(line)

		clean_line = clean_pattern.sub('', stripped_line)
		balance += clean_line.count('(') + clean_line.count('[') + clean_line.count('{')
		balance -= clean_line.count(')') + clean_line.count(']') + clean_line.count('}')

		if balance == 0 and not stripped_line.endswith('\\'):
			block_code = "\n".join(buffer)
			buffer = []

			try:
				tree = ast.parse(block_code.strip())

				for node in tree.body:
					if isinstance(node, ast.Assign) and len(node.targets) == 1:
						target = node.targets[0]
						if isinstance(target, ast.Name):
							env[target.id] = node.value

					elif isinstance(node, ast.Assert):
						res = process_assert_node(node, env)
						if res: extracted_cases.append(res)

			except Exception:
				pass

	return extracted_cases

def process_assert_node(assert_node, env):
	try:
		compare = assert_node.test
		if not isinstance(compare, ast.Compare): return None
		left_node = compare.left
		right_node = compare.comparators[0]
		if not isinstance(left_node, ast.Call): return None

		new_args = []
		for arg in left_node.args:
			if isinstance(arg, ast.Name) and arg.id in env:
				new_args.append(env[arg.id])
			else:
				new_args.append(arg)

		new_keywords = []
		for kw in left_node.keywords:
			if isinstance(kw.value, ast.Name) and kw.value.id in env:
				new_kw = ast.keyword(arg=kw.arg, value=env[kw.value.id])
				new_keywords.append(new_kw)
			else:
				new_keywords.append(kw)

		resolved_call = ast.Call(func=left_node.func, args=new_args, keywords=new_keywords)
		return f"assert {ast.unparse(resolved_call)} == {ast.unparse(right_node)}"
	except:
		return None

def extract_calls_answers(text_line):
	try:
		node = ast.parse(text_line).body[0]

		if not isinstance(node, ast.Assert):
			return None, None
		compare = node.test

		if not isinstance(compare, ast.Compare):
			return None, None

		left_node = compare.left
		right_node = compare.comparators[0]

		if not isinstance(left_node, ast.Call):
			return None, None

		full_call_str = ast.unparse(left_node)
		answer_str = ast.unparse(right_node)

		return full_call_str, answer_str

	except Exception as e:
		return None, None

def safe_run_wrapper(index: int, request: RunCodeRequest):
	try:
		result = run_code(request)
		return index, result
	except Exception as e:
		return index, e

def execute_code(code: List[str], run_timeout=10) -> List[RunCodeResponse]:
	results = [None] * len(code)
	pending_indices = list(range(len(code)))
	max_retry_times = 5

	while max_retry_times > 0 and pending_indices:
		max_retry_times -= 1
		args_list = [
		 [idx, RunCodeRequest(code=code[idx], language='python', run_timeout=run_timeout)]
		 for idx in pending_indices
		]

		try:
			batch_outputs = run_concurrent(
			 safe_run_wrapper,
			 args=args_list,
			 concurrency=20
			)

			new_pending_indices = []
			sandbox_needs_restart = False

			for output in batch_outputs:
				if not isinstance(output, tuple):
					continue
				idx, res = output

				if isinstance(res, Exception):
					print(f"Task {idx} failed with error: {res}")
					new_pending_indices.append(idx)
					sandbox_needs_restart = True
				else:
					results[idx] = res

			pending_indices = new_pending_indices

			if sandbox_needs_restart and pending_indices:
				print(f"Sandbox error detected. Retrying {len(pending_indices)} tasks after sleep...")
				time.sleep(60)

		except Exception as e:
			print(f"Concurrent error: {e}")
			time.sleep(60)

	final_output = []
	for res in results:
		if res is None:
			final_output.append(RunCodeResponse(
			 status=RunStatus.Failed,
			 message='Sandbox Error',
			 run_result=CommandRunResult(status=CommandRunStatus.TimeLimitExceeded, stderr='Sandbox Error')
			))
		else:
			final_output.append(res)

	return final_output

def extract_code_from_response(response_list: List[str]):
	code_list = []
	nonrelated_ratio_list = []
	for response in response_list:

		if "```python" in response and "```" in response.split("```python")[1]:
			code_list.append(response.split("```python")[1].split("```", 1)[0].strip('\n'))
			nonrelated_ratio_list.append((len(response.split("```python", 1)[0]) + len(response.split("```python", 1)[1].split("```", 1)[1])) / len(response))
		elif "```" in response and "```" in response.split("```")[1]:
			code_list.append(response.split("```", 1)[1].split("```", 1)[0].strip('\n'))
			nonrelated_ratio_list.append((len(response.split("```", 1)[0]) + len(response.split("```", 1)[1].split("```", 1)[1])) / len(response))
		else:
			code_list.append(response.strip('\n'))
			nonrelated_ratio_list.append(1.0)
	return code_list, {"nonrelated_ratio": nonrelated_ratio_list}

def extract_testcase_from_response(response_list: List[str]):
	"""Extract test cases from response, adapt to SFT cold start format, enhance robustness"""
	testcase_list = []
	nonprintable_ratio_list = []
	space_ratio_list = []
	nonrelated_ratio_list = []
	overlong_ratio_list = []
	for response in response_list:
		pattern = r"```python(.*?)```"
		matches = re.findall(pattern, response, re.DOTALL)
		if matches:
			testcase_content = '\n'.join(matches)
			assert_lines = extract_asserts(testcase_content)

			testcase_list.append(json.dumps({
			 "assert_statements": assert_lines

			}))
			nonrelated_ratio_list.append(1 - len(testcase_content) / len(response))

		else:
			testcase_list.append(json.dumps({"assert_statements": response.strip()}))
			nonrelated_ratio_list.append(1.0)

		nonprintable_ratio_list.append(len([c for c in response if c not in string.printable]) / len(response))
		space_ratio_list.append((response.count(' ') + response.count('\t') * 4 + response.count('\n')) / len(response))
		overlong_ratio_list.append(len(response) / 2048)

	return testcase_list, {"nonprintable_ratio": nonprintable_ratio_list, "space_ratio": space_ratio_list, "nonrelated_ratio": nonrelated_ratio_list, "overlong_ratio": overlong_ratio_list}

def validate_and_fill_generated_testcases(generated_testcase_str_list: List[str], gt_code_list: List[str]) -> Tuple[List[List[str]], List[int], List[int]]:
	"""
	Validate generated test cases and fill __TO_BE_FILLED__ placeholder

	Args:
		generated_testcase_str_list: Generated test cases string (JSON format) list
		gt_code_list: GT code list

	Returns:
		(list of valid test cases list(not repeated), list of invalid test cases number(not executable, not correct, or repeated), list of all test cases number)
	"""
	valid_tests_list = []
	invalid_count_list = []
	total_count_list = []
	test_codes = []
	gen_tests = []

	try:
		for generated_testcase_str, gt_code in zip(generated_testcase_str_list, gt_code_list):
			invalid_count = 0

			try:
				test_data = json.loads(generated_testcase_str)
			except json.JSONDecodeError as e:

				try:

					fixed_str = generated_testcase_str.replace("'", '"')

					fixed_str = re.sub(r',\s*}', '}', fixed_str)
					fixed_str = re.sub(r',\s*]', ']', fixed_str)
					test_data = json.loads(fixed_str)
				except Exception as e:
					raise Exception(f"JSON parsing also failed: {e}")

			if not isinstance(test_data, dict):
				raise Exception(f"Test data is not a dictionary: {type(test_data)}")

			if "assert_statements" not in test_data:
				raise Exception(f"Test data is missing 'assert_statements' field, available fields: {list(test_data.keys())}")

			assert_statements = test_data["assert_statements"]

			if isinstance(assert_statements, str):

				assert_statements = [line.strip() for line in assert_statements.split('\n')
				    if line.strip() and line.strip().startswith('assert')]
			elif not isinstance(assert_statements, list):
				raise Exception(f"assert_statements is not a list or string: {type(assert_statements)}")

			normalized_stmts = []
			calls = []
			answers = []
			total_count = len(assert_statements)
			for i, stmt in enumerate(assert_statements):
				if not isinstance(stmt, str):
					print(f"assert statement {i+1} is not a string: {type(stmt)}")
					invalid_count += 1
					continue

				stmt = stmt.strip()
				func_call_part, answer_part = extract_calls_answers(stmt)
				if not func_call_part or not answer_part:
					invalid_count += 1
					continue

				stmt = f"assert {func_call_part} == __TO_BE_FILLED__"

				normalized_stmts.append(stmt)
				calls.append(func_call_part)
				answers.append(answer_part)

			invalid_count_list.append(invalid_count)
			total_count_list.append(total_count)

			lines = []
			lines.append("# GT code")
			lines.append(gt_code)
			lines.append("# Generated testcases")
			for idx, (call, answer) in enumerate(zip(calls, answers)):
				lines.append(f"print('__CASE_START__{idx}')")
				lines.append("try:")
				lines.append(f"    _r = {call}")
				lines.append(f"    print('__CASE_RES__{idx}:' + repr(_r))")

				lines.append(f"    print('__CASE_VAL__{idx}:' + repr(((_r) == ({answer}))))")
				lines.append("except Exception as _e:")
				lines.append(f"    print('__CASE_ERR__{idx}:' + repr(_e))")
			test_codes.append('\n'.join(lines))

			gen_tests.append(normalized_stmts)

		results = execute_code(test_codes)

		for i, resp in enumerate(results):
			stdout = resp.run_result.stdout if (resp.status == RunStatus.Success and resp.run_result and resp.run_result.status == CommandRunStatus.Finished) else ""
			valid_tests = set()
			invalid_count = invalid_count_list[i]
			normalized_stmts = gen_tests[i]

			for idx, stmt in enumerate(normalized_stmts):
				key_res = f"__CASE_RES__{idx}:"
				key_val = f"__CASE_VAL__{idx}:"
				key_err = f"__CASE_ERR__{idx}"
				if key_res in stdout:
					line = next((ln for ln in stdout.splitlines() if key_res in ln), "")
					result_repr = line.split(key_res, 1)[1].strip()
					processed_stmt = stmt.replace("__TO_BE_FILLED__", result_repr)

					if processed_stmt in valid_tests:
						invalid_count += 1
						continue
					valid_tests.add(processed_stmt)

					line = next((ln for ln in stdout.splitlines() if key_val in ln), "")
					if not line:
						invalid_count += 1
						continue
					val_repr = line.split(key_val, 1)[1].strip()
					if val_repr == "False":
						invalid_count += 1
				elif key_err in stdout:
					invalid_count += 1
				else:
					invalid_count += 1

			valid_tests_list.append(list(valid_tests))
			invalid_count_list[i] = invalid_count

	except Exception as e:
		raise Exception(f"Validation process encountered severe error: {e}\nError details: {traceback.format_exc()}")

	return valid_tests_list, invalid_count_list, total_count_list

def evaluate(
 generated_code_list: List[str],
 generated_testcase_str_list: List[str],
 gt_code_list: List[str],
 gt_testcase_list: List[List[str]],
 attack_testcase_list: List[List[str]] = []
) -> Tuple[List[bool], List[List[str]], List[int], List[int], List[int], List[int], List[List[str]], List[int], List[int], List[List[str]], List[str], List[bool], List[bool], List[int], List[List[str]], List[List[str]]]:
	"""
	Comprehensive validation and testing function that combines sandbox execution operations.
	Preprocessing is done locally, only execution parts are merged into sandbox.

	Args:
		generated_code_list: List of generated codes to be validated and tested
		generated_testcase_str_list: List of generated test case strings (JSON format)
		gt_code_list: List of ground truth codes
		gt_testcase_list: List of ground truth test cases
		attack_testcase_list: Optional list of attack testcases for each code (List[List[str]])

	Returns:
		Tuple containing:
		- is_code_valid_list: Whether each generated code is valid
		- valid_testcase_list: List of valid test cases for each item
		- invalid_testcase_count_list: Count of invalid test cases for each item
		- total_testcase_count_list: Total count of test cases for each item
		- gt_passed_list: Number of passed GT tests for each code
		- gt_total_list: Total number of GT tests for each code
		- gt_failed_msgs_list: List of failed messages for GT tests
		- gen_passed_list: Number of passed generated tests for each code
		- gen_total_list: Total number of generated tests for each code
		- gen_failed_msgs_list: List of failed messages for generated tests
		- gen_failed_tcs_list: List of failed generated testcases for each testcase sample (List[List[str]])
		- stdout_list: List of stdout for each test
		- timeout_list: List of whether the test timed out
		- error_list: List of whether the test encountered an error
		- attack_gen_total_list: Total number of attack tests for each code
		- attack_gen_passed_tcs_list: List of passed attack testcases for each code (List[List[str]])
	"""

	assert len(generated_testcase_str_list) % len(generated_code_list) == 0, "Number of generated test cases must be a multiple of number of generated codes"
	n_samples_test = len(generated_testcase_str_list) // len(generated_code_list)

	calls_list: List[List[str]] = []
	answers_list: List[List[str]] = []
	normalized_list: List[List[str]] = []
	invalid_testcase_count_list: List[int] = []
	total_testcase_count_list: List[int] = []
	for generated_testcase_str in generated_testcase_str_list:
		invalid_count = 0

		try:
			test_data = json.loads(generated_testcase_str)
		except json.JSONDecodeError as e:
				try:
					fixed_str = generated_testcase_str.replace("'", '"')
					fixed_str = re.sub(r',\s*}', '}', fixed_str)
					fixed_str = re.sub(r',\s*]', ']', fixed_str)
					test_data = json.loads(fixed_str)
				except Exception:
					test_data = {"assert_statements": []}

		assert_statements = test_data.get("assert_statements", [])
		if isinstance(assert_statements, str):
			assert_statements = [line.strip() for line in assert_statements.split('\n') if line.strip() and line.strip().startswith('assert')]

		normalized_stmts = []
		calls = []
		answers = []
		total_count = len(assert_statements)
		for stmt in assert_statements:
			if not isinstance(stmt, str):
				invalid_count += 1
				continue
			stmt = stmt.strip()
			func_call_part, answer_part = extract_calls_answers(stmt)
			if not func_call_part or not answer_part:
				invalid_count += 1
				continue

			stmt = f"assert {func_call_part} == __TO_BE_FILLED__"
			normalized_stmts.append(stmt)
			calls.append(func_call_part)
			answers.append(answer_part)
		calls_list.append(calls)
		answers_list.append(answers)
		normalized_list.append(normalized_stmts)
		invalid_testcase_count_list.append(invalid_count)
		total_testcase_count_list.append(total_count)

	test_codes = []

	for i, (generated_code, gt_code) in enumerate(zip(generated_code_list, gt_code_list)):
		(gen_clean,), gen_imps, _ = extract_imps_fns([generated_code])
		(gt_clean,), gt_imps, gt_fns = extract_imps_fns([gt_code])
		global_imports = sorted(gen_imps | gt_imps)
		lines = ["\n".join(global_imports)]

		lines.append("valid_asserts_list = []")
		lines.append("try:")
		lines.append(textwrap.indent(gt_clean, "    "))
		for n in range(n_samples_test):
			test_idx = i * n_samples_test + n
			calls = calls_list[test_idx]
			answers = answers_list[test_idx]
			normalized_stmts = normalized_list[test_idx]
			invalid_count = invalid_testcase_count_list[test_idx]

			lines.append(f"    invalid_count = {invalid_count}")
			lines.append("    valid_asserts = []")
			lines.append("    seen = set()")
			lines.append("    gen_total = 0")
			for stmt, call, ans in zip(normalized_stmts, calls, answers):
				lines.append("    try:")
				lines.append(f"        _r = {call}")
				lines.append(f"        stmt = {repr(stmt)}.replace('__TO_BE_FILLED__', repr(_r))")
				lines.append("        if stmt in seen:")
				lines.append("            invalid_count += 1")
				lines.append("            valid_asserts.append(None)")
				lines.append("        else:")
				lines.append("            seen.add(stmt)")
				lines.append("            valid_asserts.append({'val': _r, 'stmt': stmt})")
				lines.append(f"            print('__GEN_START__{test_idx}_' + str(gen_total) + ':' + repr(stmt), flush=True)")
				lines.append("            gen_total += 1")
				lines.append(f"            if (_r) != ({ans}):")
				lines.append("                invalid_count += 1")
				lines.append("    except Exception:")
				lines.append("        invalid_count += 1")
				lines.append("        valid_asserts.append(None)")
			lines.append(f"    print('__INVALID_TEST__{test_idx}:' + str(invalid_count), flush=True)")
			lines.append("    valid_asserts_list.append(valid_asserts)")

		lines.append("except Exception:")
		lines.append("    pass")

		for gt_fn in sorted(gt_fns):
			if gt_fn:
				lines.append("try:")
				lines.append(f"    del {gt_fn}")
				lines.append("except Exception:")
				lines.append("    pass")

		try:
			ast.parse(gen_clean)
			syntax_valid = True
		except:
			syntax_valid = False

		if not syntax_valid:
			test_codes.append('\n'.join(lines))
			continue

		lines.append("try:")
		lines.append(textwrap.indent(gen_clean, "    "))
		lines.append(f"    print('__CODE_VALID__{i}', flush=True)")

		if gt_testcase_list:
			gt_testcase = gt_testcase_list[i]
			for idx, test_item in enumerate(gt_testcase):
				lines.append("    try:")
				lines.append(textwrap.indent(test_item, "        "))
				lines.append(f"        print('__GT_PASS__{i}_{idx}', flush=True)")
				lines.append("    except Exception as e:")
				lines.append(f"        print('__GT_FAIL__{i}_{idx}:' + repr(e), flush=True)")

		attack_testcases = attack_testcase_list[i] if attack_testcase_list else []
		if attack_testcases:
			for idx, test_item in enumerate(attack_testcases):
				lines.append(f"    print('__ATTACK_START__{i}_{idx}:' + repr({repr(test_item)}), flush=True)")
				lines.append("    try:")
				lines.append(f"        {test_item}")
				lines.append(f"        print('__ATTACK_PASS__{i}_{idx}', flush=True)")
				lines.append("    except Exception as e:")
				lines.append(f"        print('__ATTACK_FAIL__{i}_{idx}:' + repr(e), flush=True)")

		for n in range(n_samples_test):
			test_idx = i * n_samples_test + n
			calls = calls_list[test_idx]

			lines.append(f"    valid_asserts = valid_asserts_list[{n}]")
			lines.append("    gen_total = 0")
			for j, call in enumerate(calls):
				lines.append(f"    spec = valid_asserts[{j}]")
				lines.append(f"    if spec is not None:")
				lines.append("        try:")
				lines.append(f"            assert {call} == spec['val']")
				lines.append(f"            print('__GEN_PASS__{test_idx}_' + str(gen_total), flush=True)")
				lines.append("        except Exception as e:")
				lines.append(f"            print('__GEN_FAIL__{test_idx}_' + str(gen_total) + ':' + repr(e), flush=True)")
				lines.append(f"        gen_total += 1")

		lines.append("except Exception:")
		lines.append("    pass")

		test_codes.append('\n'.join(lines))

	results = execute_code(test_codes)

	is_code_valid_list = []
	gt_passed_list = []
	gt_total_list = []
	gt_failed_msgs_list = []
	gen_passed_list = []
	gen_total_list = []
	gen_failed_msgs_list = []
	gen_failed_tcs_list = []
	valid_testcase_list = []
	stdout_list = []
	timeout_list = []
	error_list = []
	attack_gen_total_list = []
	attack_gen_passed_tcs_list = []

	for i, resp in enumerate(results):
		output = resp.run_result.stdout if (resp.run_result and resp.run_result.stdout) else ""
		timeout = (resp.run_result.status == CommandRunStatus.TimeLimitExceeded) if (resp.run_result and resp.run_result.status) else True
		error = resp.run_result.stderr if (resp.run_result and resp.run_result.stderr) else ""
		stdout_list.append(output)
		timeout_list.append(timeout)
		error_list.append(error)

		code_valid = f'__CODE_VALID__{i}' in output
		is_code_valid_list.append(code_valid)

		if gt_testcase_list:
			gt_passed = 0
			gt_total = len(gt_testcase_list[i])
			gt_failed_msgs = []

			for idx in range(gt_total):
				if f'__GT_PASS__{i}_{idx}' in output:
					gt_passed += 1
				elif f'__GT_FAIL__{i}_{idx}:' in output:
					fail_line = next((line for line in output.splitlines() if f'__GT_FAIL__{i}_{idx}:' in line), None)
					msg = fail_line if fail_line else f"GT Test {idx+1}: Failed"
					gt_failed_msgs.append(msg)
				else:
					gt_failed_msgs.append(f"GT Test {idx+1}: No result found")

			gt_passed_list.append(gt_passed)
			gt_total_list.append(gt_total)
			gt_failed_msgs_list.append(gt_failed_msgs)

		attack_testcases = attack_testcase_list[i] if attack_testcase_list else []
		attack_passed = 0
		attack_total = len(attack_testcases)
		attack_passed_tcs = []
		if attack_total > 0:
			for idx in range(attack_total):

				start_line = next((line for line in output.splitlines() if f"__ATTACK_START__{i}_{idx}:" in line), None)
				testcase = start_line.split(f"__ATTACK_START__{i}_{idx}:")[1][1:-1] if start_line else ""
				if f'__ATTACK_PASS__{i}_{idx}' in output:
					attack_passed += 1
					attack_passed_tcs.append(testcase)
		attack_gen_total_list.append(attack_total)
		attack_gen_passed_tcs_list.append(attack_passed_tcs)

		for n in range(n_samples_test):
			test_idx = i * n_samples_test + n

			if f'__INVALID_TEST__{test_idx}:' in output:
				invalid_count_line = next((line for line in output.splitlines() if f'__INVALID_TEST__{test_idx}:' in line), None)
				invalid_testcase_count_list[test_idx] = int(invalid_count_line.split(f'__INVALID_TEST__{test_idx}:')[1]) if invalid_count_line else 0

			gen_passed = 0
			gen_total = 0
			gen_failed_msgs = []
			gen_failed_tcs = []
			valid_testcases = []
			while f"__GEN_START__{test_idx}_{gen_total}" in output:
				idx = gen_total
				test_line = next((line for line in output.splitlines() if f"__GEN_START__{test_idx}_{idx}" in line), None)
				valid_testcase = test_line.split(f"__GEN_START__{test_idx}_{idx}:")[1][1:-1] if test_line else ""
				valid_testcases.append(valid_testcase)

				if f'__GEN_PASS__{test_idx}_{idx}' in output:
					gen_passed += 1
					gen_failed_msgs.append('')
				elif f'__GEN_FAIL__{test_idx}_{idx}:' in output:
					fail_line = next((line for line in output.splitlines() if f'__GEN_FAIL__{test_idx}_{idx}:' in line), None)
					msg = fail_line if fail_line else f"Generated Test {test_idx+1}: Failed"
					gen_failed_msgs.append(msg)
					gen_failed_tcs.append(valid_testcase)
				else:
					gen_failed_msgs.append(f"Generated Test {test_idx+1}: No result found")
					gen_failed_tcs.append(valid_testcase)
				gen_total += 1

			gen_passed_list.append(gen_passed)
			gen_total_list.append(gen_total)
			gen_failed_msgs_list.append(gen_failed_msgs)
			gen_failed_tcs_list.append(gen_failed_tcs)
			valid_testcase_list.append(valid_testcases)

	return (is_code_valid_list, valid_testcase_list, invalid_testcase_count_list,
	  total_testcase_count_list, gt_passed_list, gt_total_list, gt_failed_msgs_list,
	  gen_passed_list, gen_total_list, gen_failed_msgs_list,
	  stdout_list, timeout_list, error_list,
	  attack_gen_total_list, attack_gen_passed_tcs_list,
	  gen_failed_tcs_list)

def validate_generated_codes(generated_code_list: List[str], test_list: List[str], entry_point_list: List[str], timeout: int = 10) -> Tuple[List[bool], List[str], List[bool], List[bool]]:
	try:
		assert len(test_list) == len(entry_point_list), f"{len(test_list)} != {len(entry_point_list)}"
		n_samples = len(generated_code_list) // len(test_list)
		test_codes = []

		header = textwrap.dedent(f"""
		import signal
		class TO(Exception): pass
		def handler(s, f): raise TO()
		def run_safe(func):
			try:
				signal.signal(signal.SIGALRM, handler)
				signal.alarm({timeout})
				func()
				signal.alarm(0)
				print('__PASS__', flush=True)
			except TO:
				print('__TIMEOUT__', flush=True)
			except:
				signal.alarm(0)
				print('__FAIL__', flush=True)
		""")

		for i, (test, entry_point) in enumerate(zip(test_list, entry_point_list)):
			current_gens = generated_code_list[i*n_samples : (i+1)*n_samples]
			current_gens, imports, _ = extract_imps_fns(current_gens)
			batch_code = ["\n".join(sorted(imports)) + "\n" + header + "\n" + test]

			for j, code in enumerate(current_gens):
				func_name = f"c_{j}"
				try:
					ast.parse(code)
					body = textwrap.indent(code + f"\ncheck({entry_point})", '    ')
					batch_code.append(f"def {func_name}():\n{body}")
				except SyntaxError:
					batch_code.append(f"def {func_name}(): raise Exception('SyntaxError')")

			main_calls = "\n".join([f"run_safe(c_{j})" for j in range(len(current_gens))])
			batch_code.append(main_calls)
			test_codes.append("\n".join(batch_code))

		results = execute_code(test_codes, timeout * n_samples + 5)

		pass_list, stdout_list, timeout_list, error_list = [], [], [], []
		for resp in results:
			out = resp.run_result.stdout if (resp.run_result and resp.run_result.stdout) else ""
			timeout = (resp.run_result.status == CommandRunStatus.TimeLimitExceeded) if (resp.run_result and resp.run_result.status) else True
			err = resp.run_result.stderr if (resp.run_result and resp.run_result.stderr) else ""
			lines = [line.strip() for line in out.splitlines() if line.strip() in ('__PASS__', '__TIMEOUT__', '__FAIL__')]
			batch_res = [x == '__PASS__' for x in lines]
			if len(batch_res) < n_samples:
				batch_res.extend([False] * (n_samples - len(batch_res)))

			pass_list.extend(batch_res)
			stdout_list.extend([out] + ['(see above)'] * (n_samples - 1))
			timeout_list.extend([timeout] * n_samples)
			error_list.extend([err] * n_samples)

		return pass_list, stdout_list, timeout_list, error_list

	except Exception as e:
		raise Exception(f"Error occurred while validating codes: {e}")

def compute_dual_model_reward(
 is_code_valid: bool,
 gt_passed: int,
 gt_total: int,
 gt_failed_msgs: List[str],
 valid_generated_tests: List[str],
 invalid_count: int,
 total_generated_count: int,
 gen_passed: int,
 gen_failed_msgs: List[str],
 test_phase: str = "comprehensive",
 **kwargs,
) -> Dict[str, Any]:
	"""
	Dual model collaborative training reward calculation function
	"""

	reward_a = 0.0
	reward_b = 0.0
	alpha = kwargs.get("alpha", 0.5)
	gt_default = 8

	code_nonrelated_ratio = kwargs.get("code_nonrelated_ratio", -1)
	test_nonprintable_ratio = kwargs.get("test_nonprintable_ratio", -1)
	test_space_ratio = kwargs.get("test_space_ratio", -1)
	test_nonrelated_ratio = kwargs.get("test_nonrelated_ratio", -1)
	test_overlong_ratio = kwargs.get("test_overlong_ratio", -1)
	attack_gen_passed = kwargs.get("attack_gen_passed", 0)
	attack_gen_total = kwargs.get("attack_gen_total", 0)

	reward_extra_info = {
	 "test_phase": test_phase,
	 "valid_generated_tests": valid_generated_tests,
	 "invalid_test_count": invalid_count,
	 "total_test_count": total_generated_count,
	 "invalid_test_ratio": invalid_count / total_generated_count if total_generated_count else 1,
	 "is_generated_code_valid": is_code_valid,
	 "code_nonrelated_ratio": code_nonrelated_ratio,
	 "test_nonprintable_ratio": test_nonprintable_ratio,
	 "test_space_ratio": test_space_ratio,
	 "test_nonrelated_ratio": test_nonrelated_ratio,
	 "test_overlong_ratio": test_overlong_ratio,
	}

	try:

		if gt_total != -1:
			reward_a_base_component = 0
			reward_extra_info["gt_test_results"] = {
			 "gt_passed": 0, "gt_total": gt_total, "gt_pass_rate": 0.0
			}
			if is_code_valid:

				gt_pass_rate = gt_passed / gt_total

				reward_extra_info["gt_test_results"] = {
				 "gt_passed": gt_passed, "gt_total": gt_total, "gt_pass_rate": gt_pass_rate
				}

				reward_a_base_component = gt_pass_rate
			reward_a += reward_a_base_component
			return {
			 "reward_a": reward_a,
			 "reward_extra_info": reward_extra_info,
			}

		if test_nonprintable_ratio > 0:
			reward_b += -1.0 * test_nonprintable_ratio

		valid_ratio = max(0.0, min(1.0, len(valid_generated_tests) / gt_default, 1 - invalid_count / gt_default))
		reward_extra_info["valid_reward"] = valid_ratio

		gen_total = len(valid_generated_tests)
		gen_fail_list = [1 if msg else 0 for msg in gen_failed_msgs][:gt_default]
		gen_fail_rate = sum(gen_fail_list) / gt_default
		attack_fail_rate = (attack_gen_total - attack_gen_passed) / attack_gen_total if attack_gen_total > 0 else 0

		gen_pass_rate = (gen_passed / gen_total) if gen_total > 0 else 0
		attack_pass_rate = (attack_gen_passed / attack_gen_total) if attack_gen_total > 0 else 0

		if attack_gen_total > 0 or gen_total > 0:
			if attack_gen_total > 0 and gen_total > 0:

				final_pass_rate = (attack_pass_rate + gen_pass_rate) / 2.0
				final_attack_rate = (gen_fail_rate - attack_fail_rate + 1) / 2.0
			elif attack_gen_total > 0:

				final_pass_rate = attack_pass_rate
				final_attack_rate = 0
			else:

				final_pass_rate = gen_pass_rate
				final_attack_rate = gen_fail_rate
		else:
			final_pass_rate = 0
			final_attack_rate = 0

		reward_extra_info["generated_test_results"] = {
		 "gen_passed": gen_passed, "gen_total": gen_total, "gen_pass_rate": gen_pass_rate,
		 "attack_passed": attack_gen_passed, "attack_total": attack_gen_total, "attack_pass_rate": attack_pass_rate,
		 "final_pass_rate": final_pass_rate, "final_attack_rate": final_attack_rate
		}

		reward_a += final_pass_rate
		reward_b += (valid_ratio * alpha + final_attack_rate * (1 - alpha)) if is_code_valid else 0

	except Exception as e:
		raise Exception(f"Error occurred while computing dual model reward: {e}")

	return {
	 "reward_a": reward_a,
	 "reward_b": reward_b,
	 "reward_extra_info": reward_extra_info,
	}

@register("acg_gt")
class ACGRewardManager:
	"""The reward manager."""

	def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs) -> None:
		"""
		Initialize the ACGRewardManager instance.

		Args:
			tokenizer: The tokenizer used to decode token IDs into text.
			num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
			compute_score: A function to compute the reward score.
			reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
		"""
		self.tokenizer = tokenizer
		self.num_examine = num_examine
		self.compute_score = compute_dual_model_reward
		self.reward_fn_key = reward_fn_key
		self.reward_kwargs = reward_kwargs

	def __call__(self, data: DataProto, is_last_step = False):
		if data.non_tensor_batch[self.reward_fn_key].tolist()[0] == "kodcode_light":

			generated_code_list, generated_code_info = extract_code_from_response(self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True))
			gt_code_list = data.non_tensor_batch["GT_code"].tolist()
			gt_testcase_list = data.non_tensor_batch["GT_test"].tolist()

			is_code_valid_list,\
   valid_testcase_list_phase2, invalid_testcase_count_phase2, total_testcase_count_phase2,\
   gt_passed, gt_total, gt_failed_msgs,\
   gen_adv_passed_phase2, gen_adv_total_phase2, gen_adv_failed_msgs_phase2,\
   stdout_list, timeout_list, error_list,\
   attack_gen_total, attack_gen_passed_tcs,\
   gen_failed_tcs_phase2 = evaluate(
			 generated_code_list,
			 [],
			 gt_code_list,
			 gt_testcase_list
			)

			reward_a_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.bfloat16)
			reward_extra_infos_dict = defaultdict(list)
			for stdout, timeout, error in zip(stdout_list, timeout_list, error_list):
				reward_extra_infos_dict['stdout'].append(stdout)
				reward_extra_infos_dict['timeout'].append(timeout)
				reward_extra_infos_dict['error'].append(error)
			sandbox_metrics = {}
			sandbox_metrics['sandbox/total'] = len(error_list)
			sandbox_metrics['sandbox/timeout_count'] = sum(timeout_list)
			sandbox_metrics['sandbox/timeout_ratio'] = sum(timeout_list) / len(timeout_list)

			for i in range(len(data)):
				data_item = data[i]

				prompt_ids = data_item.batch["prompts"]
				prompt_length = prompt_ids.shape[-1]
				valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

				try:
					reward_result_phase2 = self.compute_score(
					 is_code_valid=is_code_valid_list[i],
					 gt_passed=gt_passed[i],
					 gt_total=gt_total[i],
					 gt_failed_msgs=gt_failed_msgs[i],
					 valid_generated_tests=[],
					 invalid_count=0,
					 total_generated_count=0,
					 gen_passed=0,
					 gen_failed_msgs=[],
					 test_phase="comprehensive",
					 code_nonrelated_ratio=generated_code_info["nonrelated_ratio"][i],
					 **self.reward_kwargs
					)
				except Exception as e:
					raise Exception(f"[REWARD-ERROR] phase2 (a_idx={i}): {e}")

				reward_a = reward_result_phase2["reward_a"]

				for key, value in reward_result_phase2.items():
					reward_extra_infos_dict[key].append(value)

				reward_a_tensor[i, valid_response_length - 1] = reward_a

			reward_extra_infos = reward_extra_infos_dict.get('reward_extra_info', [])
			gt_pass_rate = np.mean([reward_extra_info.get('gt_test_results', {}).get('gt_pass_rate', 0) for reward_extra_info in reward_extra_infos])
			valid_code = np.mean([reward_extra_info.get('is_generated_code_valid', False) for reward_extra_info in reward_extra_infos])
			sandbox_metrics['sandbox/gt_pass_rate'] = gt_pass_rate
			sandbox_metrics['sandbox/valid_code'] = valid_code

			return {
			 "reward_a_tensor": reward_a_tensor,
			 "reward_extra_infos_dict": reward_extra_infos_dict,
			 "sandbox_metrics": sandbox_metrics,
			}
		else:
			generated_code_list, generated_code_info = extract_code_from_response(self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True))
			test_list, entry_point_list = map(list, zip(*dict(zip(
			 data.non_tensor_batch["test"].tolist(),
			 [x["function_name"] for x in data.non_tensor_batch["extra_info"]]
			)).items()))
			pass_results, stdout_list, timeout_list, error_list = validate_generated_codes(generated_code_list, test_list, entry_point_list, 20 if is_last_step else 10)
			sandbox_infos_dict = defaultdict(list)
			for stdout, timeout, error in zip(stdout_list, timeout_list, error_list):
				sandbox_infos_dict['stdout'].append(stdout)
				sandbox_infos_dict['timeout'].append(timeout)
				sandbox_infos_dict['error'].append(error)
			sandbox_metrics = {}
			sandbox_metrics['sandbox/total'] = len(error_list)
			sandbox_metrics['sandbox/timeout_count'] = sum(timeout_list)
			sandbox_metrics['sandbox/timeout_ratio'] = sum(timeout_list) / len(timeout_list)

			return {
			 "pass_results": pass_results,
			 "sandbox_infos_dict": sandbox_infos_dict,
			 "sandbox_metrics": sandbox_metrics,
			}

def get_attack_testcases_from_mistake_book(mistake_book, question_ids: List[str], n_samples_a: int, gt_tc_counts: List[int] = None) -> List[List[str]]:
	if mistake_book is None:
		print("Mistake Book not used.")
		return []
	if gt_tc_counts is None:
		gt_tc_counts = [8] * len(question_ids) * n_samples_a
	attack_testcase_list = []
	for qidx, qid in enumerate(question_ids):
		attack_tcs = mistake_book.get_attack_testcases(qid, max_count=gt_tc_counts[qidx * n_samples_a])

		attack_testcase_list.extend([attack_tcs] * n_samples_a)
	return attack_testcase_list

def update_mistake_book(mistake_book, question_ids: List[str], n_samples_a: int, n_samples_b: int,
        generated_code_list: List[str], is_code_valid_list: List[bool],
        attack_gen_total: List[int], attack_gen_passed_tcs: List[List[str]],
        gen_adv_total_phase2: List[int], gen_adv_passed_phase2: List[int],
        gen_failed_tcs_phase2: List[List[str]]):
	if mistake_book is None:
		return

	for qidx, qid in enumerate(question_ids):
		start_code_idx = qidx * n_samples_a
		end_code_idx = (qidx + 1) * n_samples_a

		all_pass_rates = []
		new_attack_testcases = []
		old_attack_testcases_passed = []

		for code_idx in range(start_code_idx, end_code_idx):
			if is_code_valid_list[code_idx]:

				gen_total = 0
				gen_passed = 0
				attack_total = attack_gen_total[code_idx]
				attack_passed = len(attack_gen_passed_tcs[code_idx])

				old_attack_testcases_passed.extend(attack_gen_passed_tcs[code_idx])

				start_tc_idx = code_idx * n_samples_b
				end_tc_idx = (code_idx + 1) * n_samples_b
				for tc_idx in range(start_tc_idx, end_tc_idx):
					gen_total += gen_adv_total_phase2[tc_idx]
					gen_passed += gen_adv_passed_phase2[tc_idx]
					failed_testcases = gen_failed_tcs_phase2[tc_idx]
					new_attack_testcases.extend(failed_testcases)

				gen_pass_rate = (gen_passed / gen_total) if gen_total > 0 else 0
				attack_pass_rate = (attack_passed / attack_total) if attack_total > 0 else 0
				final_pass_rate = ((attack_pass_rate + gen_pass_rate) / 2.0) if attack_total > 0 else gen_pass_rate
				all_pass_rates.append(final_pass_rate)
			else:
				all_pass_rates.append(0)

		avg_pass_rate = sum(all_pass_rates) / len(all_pass_rates) if all_pass_rates else 0
		mistake_book.update(qid, new_attack_testcases, old_attack_testcases_passed, avg_pass_rate)

@register("acg_llm")
class ACGRewardManager:
	"""The reward manager."""

	def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs) -> None:
		"""
		Initialize the ACGRewardManager instance.

		Args:
			tokenizer: The tokenizer used to decode token IDs into text.
			num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
			compute_score: A function to compute the reward score.
			reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
		"""
		self.tokenizer_a, self.tokenizer_b = tokenizer
		self.num_examine = num_examine
		self.compute_score = compute_dual_model_reward
		self.reward_fn_key = reward_fn_key
		self.reward_kwargs = reward_kwargs
		self.mistake_book = None

	def __call__(self, a_data: DataProto, b2_data: DataProto = None, is_last_step = False):
		if a_data.non_tensor_batch[self.reward_fn_key].tolist()[0] == "kodcode_light":

			generated_code_list, generated_code_info = extract_code_from_response(self.tokenizer_a.batch_decode(a_data.batch["responses"], skip_special_tokens=True))
			generated_testcase_list_phase2, generated_testcase_info_phase2 = extract_testcase_from_response(self.tokenizer_b.batch_decode(b2_data.batch["responses"], skip_special_tokens=True))
			gt_code_list = a_data.non_tensor_batch["GT_code"].tolist()

			question_ids = list(dict.fromkeys([info.get("question_id", "") for info in a_data.non_tensor_batch.get("extra_info", [])]))
			n_samples_a = len(generated_code_list) // len(question_ids)
			attack_testcase_list = get_attack_testcases_from_mistake_book(
			 self.mistake_book, question_ids, n_samples_a
			)

			is_code_valid_list,\
   valid_testcase_list_phase2, invalid_testcase_count_phase2, total_testcase_count_phase2,\
   gt_passed, gt_total, gt_failed_msgs,\
   gen_adv_passed_phase2, gen_adv_total_phase2, gen_adv_failed_msgs_phase2,\
   stdout_list, timeout_list, error_list,\
   attack_gen_total, attack_gen_passed_tcs,\
   gen_failed_tcs_phase2 = evaluate(
			 generated_code_list,
			 generated_testcase_list_phase2,
			 gt_code_list,
			 [],
			 attack_testcase_list
			)

			tmp_reward_a_list = []
			reward_a_tensor = torch.zeros_like(a_data.batch["responses"], dtype=torch.bfloat16)
			reward_b2_tensor = torch.zeros_like(b2_data.batch["responses"], dtype=torch.bfloat16)
			reward_extra_infos_dict = defaultdict(list)
			sandbox_infos_dict = defaultdict(list)
			for stdout, timeout, error in zip(stdout_list, timeout_list, error_list):
				sandbox_infos_dict['stdout'].append(stdout)
				sandbox_infos_dict['timeout'].append(timeout)
				sandbox_infos_dict['error'].append(error)
			sandbox_metrics = {}
			sandbox_metrics['sandbox/total'] = len(error_list)
			sandbox_metrics['sandbox/timeout_count'] = sum(timeout_list)
			sandbox_metrics['sandbox/timeout_ratio'] = sum(timeout_list) / len(timeout_list)

			for b2_idx in range(len(b2_data)):

				a_idx = b2_idx // (len(b2_data) // len(a_data))
				data_a_item = a_data[a_idx]
				data_b2_item = b2_data[b2_idx]

				prompt_ids_a = data_a_item.batch["prompts"]
				prompt_length_a = prompt_ids_a.shape[-1]
				valid_response_length_a = data_a_item.batch["attention_mask"][prompt_length_a:].sum()
				prompt_ids_b2 = data_b2_item.batch["prompts"]
				prompt_length_b2 = prompt_ids_b2.shape[-1]
				valid_response_length_b2 = data_b2_item.batch["attention_mask"][prompt_length_b2:].sum()

				try:
					reward_result_phase2 = self.compute_score(
					 is_code_valid=is_code_valid_list[a_idx],
					 gt_passed=-1,
					 gt_total=-1,
					 gt_failed_msgs=[],
					 valid_generated_tests=valid_testcase_list_phase2[b2_idx],
					 invalid_count=invalid_testcase_count_phase2[b2_idx],
					 total_generated_count=total_testcase_count_phase2[b2_idx],
					 gen_passed=gen_adv_passed_phase2[b2_idx],
					 gen_failed_msgs=gen_adv_failed_msgs_phase2[b2_idx],
					 test_phase="adversarial",
					 test_nonprintable_ratio=generated_testcase_info_phase2["nonprintable_ratio"][b2_idx],
					 test_space_ratio=generated_testcase_info_phase2["space_ratio"][b2_idx],
					 test_nonrelated_ratio=generated_testcase_info_phase2["nonrelated_ratio"][b2_idx],
					 test_overlong_ratio=generated_testcase_info_phase2["overlong_ratio"][b2_idx],
					 code_nonrelated_ratio=generated_code_info["nonrelated_ratio"][a_idx],
					 attack_gen_passed=len(attack_gen_passed_tcs[a_idx]) if attack_gen_passed_tcs else 0,
					 attack_gen_total=attack_gen_total[a_idx] if attack_gen_total else 0,
					 **self.reward_kwargs
					)
				except Exception as e:
					raise Exception(f"[REWARD-ERROR] phase2 (a_idx={a_idx}, b2_idx={b2_idx}): {e}")

				reward_a = reward_result_phase2["reward_a"]
				reward_b2 = reward_result_phase2["reward_b"]

				for key, value in reward_result_phase2.items():
					reward_extra_infos_dict[key].append(value)

				tmp_reward_a_list.append(reward_a)
				if len(tmp_reward_a_list) == len(b2_data) // len(a_data):
					reward_a_tensor[a_idx, valid_response_length_a - 1] = sum(tmp_reward_a_list) / len(tmp_reward_a_list)
					tmp_reward_a_list = []

				reward_b2_tensor[b2_idx, valid_response_length_b2 - 1] = reward_b2

			n_samples_b = len(generated_testcase_list_phase2) // len(generated_code_list)
			update_mistake_book(
			 self.mistake_book,
			 question_ids=question_ids,
			 n_samples_a=n_samples_a,
			 n_samples_b=n_samples_b,
			 generated_code_list=generated_code_list,
			 is_code_valid_list=is_code_valid_list,
			 attack_gen_total=attack_gen_total,
			 attack_gen_passed_tcs=attack_gen_passed_tcs,
			 gen_adv_total_phase2=gen_adv_total_phase2,
			 gen_adv_passed_phase2=gen_adv_passed_phase2,
			 gen_failed_tcs_phase2=gen_failed_tcs_phase2
			)

			reward_extra_infos = reward_extra_infos_dict.get('reward_extra_info', [])
			valid_test = np.mean([reward_extra_info.get('generated_test_results', {}).get('gen_total', 0) for reward_extra_info in reward_extra_infos])
			gen_pass = np.mean([reward_extra_info.get('generated_test_results', {}).get('gen_passed', 0) for reward_extra_info in reward_extra_infos])
			gen_pass_rate = np.mean([reward_extra_info.get('generated_test_results', {}).get('gen_pass_rate', 0) for reward_extra_info in reward_extra_infos])
			attack_pass_rate = np.mean([reward_extra_info.get('generated_test_results', {}).get('attack_pass_rate', 0) for reward_extra_info in reward_extra_infos])
			final_pass_rate = np.mean([reward_extra_info.get('generated_test_results', {}).get('final_pass_rate', 0) for reward_extra_info in reward_extra_infos])
			valid_code = np.mean([reward_extra_info.get('is_generated_code_valid', False) for reward_extra_info in reward_extra_infos])
			total_test = np.mean([reward_extra_info.get('total_test_count', 0) for reward_extra_info in reward_extra_infos])
			invalid_test = np.mean([reward_extra_info.get('invalid_test_count', 0) for reward_extra_info in reward_extra_infos])
			invalid_ratio = np.mean([reward_extra_info.get('invalid_test_ratio', 0) for reward_extra_info in reward_extra_infos])
			sandbox_metrics['sandbox/valid_test'] = valid_test
			sandbox_metrics['sandbox/gen_pass'] = gen_pass
			sandbox_metrics['sandbox/gen_pass_rate'] = gen_pass_rate
			sandbox_metrics['sandbox/attack_pass_rate'] = attack_pass_rate
			sandbox_metrics['sandbox/final_pass_rate'] = final_pass_rate
			sandbox_metrics['sandbox/valid_code'] = valid_code
			sandbox_metrics['sandbox/total_test'] = total_test
			sandbox_metrics['sandbox/invalid_test'] = invalid_test
			sandbox_metrics['sandbox/invalid_ratio'] = invalid_ratio

			return {
			 "reward_a_tensor": reward_a_tensor,
			 "reward_b2_tensor": reward_b2_tensor,
			 "reward_extra_infos_dict": reward_extra_infos_dict,
			 "sandbox_infos_dict": sandbox_infos_dict,
			 "sandbox_metrics": sandbox_metrics,
			}

		else:
			generated_code_list, generated_code_info = extract_code_from_response(self.tokenizer_a.batch_decode(a_data.batch["responses"], skip_special_tokens=True))
			test_list, entry_point_list = map(list, zip(*dict(zip(
			 a_data.non_tensor_batch["test"].tolist(),
			 [x["function_name"] for x in a_data.non_tensor_batch["extra_info"]]
			)).items()))
			pass_results, stdout_list, timeout_list, error_list = validate_generated_codes(generated_code_list, test_list, entry_point_list, 20 if is_last_step else 10)
			sandbox_infos_dict = defaultdict(list)
			for stdout, timeout, error in zip(stdout_list, timeout_list, error_list):
				sandbox_infos_dict['stdout'].append(stdout)
				sandbox_infos_dict['timeout'].append(timeout)
				sandbox_infos_dict['error'].append(error)
			sandbox_metrics = {}
			sandbox_metrics['sandbox/total'] = len(error_list)
			sandbox_metrics['sandbox/timeout_count'] = sum(timeout_list)
			sandbox_metrics['sandbox/timeout_ratio'] = sum(timeout_list) / len(timeout_list)

			return {
			 "pass_results": pass_results,
			 "sandbox_infos_dict": sandbox_infos_dict,
			 "sandbox_metrics": sandbox_metrics,
			}
