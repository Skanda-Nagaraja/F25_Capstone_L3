#!/usr/bin/env python3
"""
Adaptive QA Agent - Self-correcting with feedback loops

This agent:
1. Scans for vulnerabilities
2. For each vulnerability:
   - Gets LLM suggestion
   - Applies fix
   - Rescans to verify
   - IF FAILED: Feeds error back to LLM and tries different approach
   - Iterates until success or max attempts
3. Learns from failures and adapts strategy
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import subprocess
import shlex
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm
from dotenv import load_dotenv

from schemas import Vulnerability, RemediationSuggestion
from openscap_cli import OpenSCAPScanner
from parse_openscap import parse_openscap
from remediation_bridge import RemediationBridge
from qa_loop import AnsibleExecutor
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

console = Console()

load_dotenv()


class AdaptiveQAAgent:
    """Self-correcting agent with feedback loops"""
    
    def __init__(self, scanner: OpenSCAPScanner, ansible_inventory: str,
                 work_dir: Path, scan_profile: str, scan_datastream: str,
                 sudo_password: Optional[str] = None, max_attempts: int = 5):
        self.scanner = scanner
        self.ansible_executor = AnsibleExecutor(ansible_inventory)
        self.ssh_executor = SSHExecutor(
            host=scanner.target_host,
            user=scanner.ssh_user,
            key=scanner.ssh_key,
            port=scanner.ssh_port,
            sudo_password=sudo_password
        )
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        self.scan_profile = scan_profile
        self.scan_datastream = scan_datastream
        self.sudo_password = sudo_password
        self.max_attempts = max_attempts
        
        # Initialize LLM agent for adaptive remediation
        self.llm_agent = self._init_llm_agent()
        
        # Track results
        self.results = {
            'fixed_first_try': [],
            'fixed_after_retry': [],
            'failed_all_attempts': [],
            'skipped': []
        }
        
        # Learning: track what works
        self.success_patterns = []
        self.failure_patterns = []
    
    def _init_llm_agent(self):
        """Initialize adaptive LLM agent"""
        # Load environment variables
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file!")
        
        # Configure OpenRouter model using OpenAIProvider
        model = OpenAIChatModel(
            model_name=os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini'),
            provider=OpenAIProvider(
                base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
                api_key=api_key
            )
        )
        
        agent = Agent(
            model,
            output_type=NativeOutput(RemediationSuggestion, strict=True),
            system_prompt=(
                "You are an adaptive security remediation agent. "
                "When a fix fails, you learn from the error and suggest a different approach. "
                "Analyze error messages, consider alternative methods, and try different strategies. "
                "Your goal is to successfully fix security vulnerabilities, adapting your approach based on feedback."
            )
        )
        return agent
    
    def scan_for_vulnerability(self, vuln: Vulnerability) -> bool:
        """Check if a specific vulnerability still exists.
        
        Returns True if vulnerability still exists, False if fixed.
        Verification is performed by rescanning and checking the specific
        OpenSCAP rule result for this vulnerability.
        """
        console.print(f"[cyan]🔍 Checking if {vuln.id} is fixed...[/cyan]")
        
        # Run scan
        scan_file = self.work_dir / f"verify_{vuln.id}.xml"
        parsed_file = self.work_dir / f"verify_{vuln.id}.json"
        
        success = self.scanner.run_scan(
            profile=self.scan_profile,
            output_file=f"/tmp/verify_{vuln.id}.xml",
            datastream=self.scan_datastream,
            sudo_password=self.sudo_password
        )
        
        if not success:
            console.print("[yellow]⚠ Could not verify, assuming not fixed[/yellow]")
            return True  # Assume still exists if scan fails
        
        # Download and parse
        self.scanner.download_results(f"/tmp/verify_{vuln.id}.xml", str(scan_file))
        parse_openscap(str(scan_file), str(parsed_file))
        
        # Check if vulnerability still exists
        with open(parsed_file) as f:
            current_vulns = json.load(f)
        
        # Look for this specific vulnerability by matching rule ID exactly
        still_exists = False
        for finding in current_vulns:
            if finding.get('rule') == vuln.title:
                still_exists = finding.get('result') in ['fail', 'error']
                break
        
        return still_exists
    
    def get_initial_remediation(self, vuln: Vulnerability) -> RemediationSuggestion:
        """Get initial remediation suggestion from LLM"""
        console.print("[cyan]🤖 Getting initial remediation from AI...[/cyan]")
        
        # Parse the OpenSCAP rule name to understand what it's checking
        rule_name = vuln.title.replace('xccdf_org.ssgproject.content_rule_', '')

        # Try deterministic, rule-based remediation first
        builtin = self.get_rule_based_remediation(vuln, rule_name)
        if builtin is not None:
            console.print("[green]Using built-in remediation for this rule[/green]")
            return builtin
        
        prompt = f"""You are remediating an OpenSCAP security compliance finding on Rocky Linux 10 (RHEL-based).

VULNERABILITY DETAILS:
- Rule: {rule_name}
- Full Rule ID: {vuln.title}
- Severity: {vuln.severity} (0=info, 1=low, 2=medium, 3=high, 4=critical)
- Host: {vuln.host}

SYSTEM INFORMATION:
- OS: Rocky Linux 10 (RHEL-based, uses dnf/yum, systemd)
- Package Manager: dnf (NOT apt)
- Init System: systemd

TASK:
Based on the rule name "{rule_name}", determine what OpenSCAP is checking and provide the EXACT commands needed to fix it.

IMPORTANT RULES:
1. Rocky Linux uses DNF, not apt-get or apt
2. Configuration files are usually in /etc/
3. For package rules: use "dnf install -y <package>"
4. For service rules: use "systemctl enable/start <service>"
5. For file permission rules: use chmod/chown
6. For audit rules: modify /etc/audit/auditd.conf or add rules to /etc/audit/rules.d/
7. For kernel parameters: modify /etc/sysctl.conf or /etc/sysctl.d/
8. For file integrity (aide): Install AND initialize with "aide --init && cp /var/lib/aide/aide.db.new.gz /var/lib/aide/aide.db.gz"
9. Be SPECIFIC - don't just install packages, configure them properly
10. DO NOT reference any 'aide.service' systemd unit (it does not exist)

Provide the EXACT commands that will make this OpenSCAP check pass.
"""
        
        result = self.llm_agent.run_sync(prompt)
        return result.output

    def get_rule_based_remediation(self, vuln: Vulnerability, rule_name: str) -> Optional[RemediationSuggestion]:
        """Return a deterministic remediation for known OpenSCAP rules.

        This improves reliability by applying well-known fixes on RHEL/Rocky.
        Returns None if no built-in remediation is available.
        """
        rn = rule_name
        cmds: List[str] = []
        notes = ""

        # AIDE rules
        if rn == 'package_aide_installed' or 'aide_installed' in rn:
            cmds = [
                'dnf install -y aide',
            ]
            notes = 'Install AIDE using dnf on Rocky/RHEL.'
        elif rn == 'aide_build_database' or 'aide_build' in rn or 'aide_init' in rn:
            cmds = [
                'dnf install -y aide',
                'aide --init',
                'if [ -f /var/lib/aide/aide.db.new.gz ]; then cp -f /var/lib/aide/aide.db.new.gz /var/lib/aide/aide.db.gz; fi'
            ]
            notes = 'Initialize AIDE database after installation.'
        elif 'aide_check_audit_tools' in rn:
            cmds = [
                'cp -n /etc/aide.conf /etc/aide.conf.bak || true',
                "grep -qE '^/sbin/audit\\* ' /etc/aide.conf || echo '/sbin/audit* p+i+n+u+g+s+m+c+sha256' >> /etc/aide.conf",
                "grep -qE '^/usr/sbin/audit\\* ' /etc/aide.conf || echo '/usr/sbin/audit* p+i+n+u+g+s+m+c+sha256' >> /etc/aide.conf",
                'aide --init',
                'if [ -f /var/lib/aide/aide.db.new.gz ]; then cp -f /var/lib/aide/aide.db.new.gz /var/lib/aide/aide.db.gz; fi',
                'aide --check'
            ]
            notes = 'Ensure audit tools are monitored by AIDE and reinitialize database.'

        # Auditd rules
        elif rn.startswith('package_audit') or rn.startswith('package_auditd'):
            cmds = [
                'dnf install -y audit',
            ]
            notes = 'Install audit package.'
        elif rn.startswith('service_auditd_enabled') or 'auditd_service' in rn:
            cmds = [
                'systemctl enable auditd',
                'systemctl start auditd'
            ]
            notes = 'Ensure auditd is enabled and running.'
        elif 'auditd_data_retention_space_left_action' in rn:
            cmds = [
                "sed -ri 's/^[[:space:]]*space_left_action[[:space:]]*=.*/space_left_action = email/' /etc/audit/auditd.conf",
                "sed -ri 's/^[[:space:]]*action_mail_acct[[:space:]]*=.*/action_mail_acct = root/' /etc/audit/auditd.conf",
                'systemctl restart auditd'
            ]
            notes = 'Configure auditd retention action and restart.'

        # Firewalld rules
        elif rn == 'package_firewalld_installed' or 'firewalld_installed' in rn:
            cmds = [
                'dnf install -y firewalld',
            ]
            notes = 'Install firewalld.'
        elif rn == 'service_firewalld_enabled' or 'firewalld_enabled' in rn:
            cmds = [
                'systemctl enable firewalld',
                'systemctl start firewalld'
            ]
            notes = 'Enable and start firewalld.'

        # SSH rules examples
        elif 'sshd_disable_root_login' in rn or 'permitrootlogin' in rn:
            cmds = [
                "if grep -qi '^[[:space:]]*PermitRootLogin' /etc/ssh/sshd_config; then sed -ri 's/^[[:space:]]*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config; else echo 'PermitRootLogin no' >> /etc/ssh/sshd_config; fi",
                'systemctl restart sshd'
            ]
            notes = 'Disable SSH root login and restart sshd.'

        # Sysctl generic pattern example
        elif rn.startswith('sysctl_'):
            # Attempt to infer key=value from rule name is unreliable; let LLM handle unless specified
            cmds = []

        if cmds:
            return RemediationSuggestion(id=vuln.id, proposed_commands=cmds, notes=notes)
        return None
    
    def get_adaptive_remediation(self, vuln: Vulnerability, 
                                 previous_attempts: List[Dict],
                                 error_message: str) -> RemediationSuggestion:
        """Get adaptive remediation based on previous failures"""
        console.print("[yellow]🔄 Getting adaptive remediation (learning from failure)...[/yellow]")
        
        # Build context from previous attempts
        attempt_history = "\n\n".join([
            f"Attempt {i+1}:\n"
            f"Commands: {', '.join(att['commands'])}\n"
            f"Ansible Execution: {'✓ Succeeded' if att['apply_success'] else '✗ Failed'}\n"
            f"OpenSCAP Verification: {'✓ FIXED' if att.get('verified', False) else '✗ STILL VULNERABLE'}\n"
            f"Output: {(att.get('error') or 'Commands executed successfully')[:500]}"
            for i, att in enumerate(previous_attempts)
        ])
        
        rule_name = vuln.title.replace('xccdf_org.ssgproject.content_rule_', '')
        
        prompt = f"""PREVIOUS REMEDIATION ATTEMPT FAILED! Analyze what went wrong and try a COMPLETELY DIFFERENT approach.

SYSTEM: Rocky Linux 10 (RHEL-based, uses dnf, systemd)

VULNERABILITY:
- Rule: {rule_name}
- OpenSCAP Rule ID: {vuln.title}
- Current Status: STILL VULNERABLE after {len(previous_attempts)} attempt(s)

WHAT HAPPENED:
{attempt_history}

ANALYSIS REQUIRED:
1. WHY did the previous approach fail?
2. Was the package/service actually installed/configured?
3. Did we miss a configuration step?
4. Is there a different way to achieve the same compliance?

COMMON OPENSCAP ISSUES:
- "package_*_installed" rules: Package installed but not configured
- "aide_*" rules: AIDE needs initialization: aide --init && mv /var/lib/aide/aide.db.new.gz /var/lib/aide/aide.db.gz
- "auditd_*" rules: Need to edit /etc/audit/auditd.conf AND restart auditd service
- "service_*" rules: Service must be both enabled AND started
- "sysctl_*" rules: Must set value AND persist it: sysctl -w key=value && echo "key=value" >> /etc/sysctl.d/99-custom.conf
- "file_permissions_*" rules: Check exact permissions required (usually mode 0600 or 0644)
- "grub2_*" rules: Must run grub2-mkconfig after editing
 - DO NOT use 'systemctl ... aide.service' (no such unit)

YOUR TASK:
Suggest a DIFFERENT strategy that addresses why the previous attempt failed.
Be MORE SPECIFIC and COMPLETE than before.
"""
        
        result = self.llm_agent.run_sync(prompt)
        return result.output
    
    def _format_sudo_command(self, command: str) -> str:
        """Wrap commands with sudo password when needed."""
        if not self.sudo_password:
            return command
        stripped = command.strip()
        if stripped.startswith("sudo "):
            without_sudo = stripped[len("sudo "):]
            return f"echo {shlex.quote(self.sudo_password)} | sudo -S {without_sudo}"
        # Default: run with sudo when password is provided
        return f"echo {shlex.quote(self.sudo_password)} | sudo -S {stripped}"

    def _write_commands_file(self, vuln: Vulnerability, attempt_num: int, commands: List[str]) -> Path:
        cmds_path = self.work_dir / f"fix_{vuln.id}_attempt{attempt_num}.cmds.txt"
        lines = []
        lines.append(f"# Commands for {vuln.id} attempt {attempt_num}\n")
        for i, cmd in enumerate(commands, 1):
            lines.append(f"{i}. {cmd}\n")
        cmds_path.write_text("".join(lines))
        return cmds_path

    def apply_remediation(self, vuln: Vulnerability, 
                         remediation: RemediationSuggestion,
                         attempt_num: int) -> Tuple[bool, str]:
        """Apply remediation and return (success, error_message)"""
        console.print(f"[cyan]🔧 Applying remediation (Attempt {attempt_num}/{self.max_attempts})...[/cyan]")
        
        # Show what we're doing
        console.print("\n[yellow]Commands to execute:[/yellow]")
        for i, cmd in enumerate(remediation.proposed_commands, 1):
            console.print(f"  {i}. {cmd}")

        # Save commands file (filter out invalid aide.service operations)
        filtered_cmds: List[str] = []
        for c in remediation.proposed_commands:
            lc = c.lower()
            if "systemctl" in lc and "aide" in lc:
                continue
            filtered_cmds.append(c)
        if not filtered_cmds:
            filtered_cmds = remediation.proposed_commands
        cmds_file = self._write_commands_file(vuln, attempt_num, filtered_cmds)
        console.print(f"\n[blue]Commands file:[/blue] {cmds_file}")

        # Execute commands directly over SSH
        success, combined_output = self.ssh_executor.execute_commands(
            [self._format_sudo_command(c) for c in filtered_cmds]
        )

        # Save log
        log_file = self.work_dir / f"fix_{vuln.id}_attempt{attempt_num}.ssh.log"
        log_file.write_text(combined_output)

        # Show output clearly
        console.print("\n[magenta]Remote Output (Attempt {}/{}):[/magenta]".format(attempt_num, self.max_attempts))
        if combined_output:
            console.print(combined_output, markup=False)
        else:
            console.print("[dim]<no output>[/dim]")

        return success, combined_output

    def process_vulnerability_adaptively(self, vuln: Vulnerability) -> Dict:
        """Process a vulnerability with adaptive retries
        
        Returns dict with results of all attempts
        """
        console.print("\n" + "="*70)
        console.print(f"[bold cyan]Processing: {vuln.title}[/bold cyan]")
        console.print("="*70 + "\n")
        
        # Show vulnerability details
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("ID", vuln.id)
        table.add_row("Title", vuln.title)
        table.add_row("Severity", f"[{'red' if int(vuln.severity) >= 3 else 'yellow'}]{vuln.severity}[/]")
        table.add_row("Host", vuln.host)
        console.print(table)
        
        attempts = []
        
        for attempt_num in range(1, self.max_attempts + 1):
            console.print(f"\n[bold yellow]═══ Attempt {attempt_num}/{self.max_attempts} ═══[/bold yellow]\n")
            
            # Get remediation suggestion
            if attempt_num == 1:
                # First attempt: get initial suggestion
                remediation = self.get_initial_remediation(vuln)
            else:
                # Subsequent attempts: adaptive based on previous failures
                last_error = attempts[-1].get('error', 'Unknown error')
                remediation = self.get_adaptive_remediation(vuln, attempts, last_error)
            
            # Show remediation
            console.print("\n[green]💡 Remediation Plan:[/green]")
            for i, cmd in enumerate(remediation.proposed_commands, 1):
                console.print(f"  {i}. [yellow]{cmd}[/yellow]")
            if remediation.notes:
                console.print(f"\n[dim]Notes: {remediation.notes}[/dim]")
            
            # Apply remediation
            time.sleep(1)  # Brief pause for readability
            apply_success, output = self.apply_remediation(vuln, remediation, attempt_num)
            
            # Record attempt
            attempt_record = {
                'attempt': attempt_num,
                'commands': remediation.proposed_commands,
                'apply_success': apply_success,
                'error': output if not apply_success else None
            }
            
            if not apply_success:
                console.print("[red]✗ Playbook execution failed[/red]")
                attempts.append(attempt_record)
                
                # Show error and ask if should continue
                if attempt_num < self.max_attempts:
                    console.print("\n[yellow]Will retry with different approach...[/yellow]")
                    time.sleep(2)
                continue
            
            console.print("[green]✓ Playbook executed successfully[/green]")
            
            # Wait for changes to take effect
            console.print("\n[cyan]⏳ Waiting 10 seconds for changes to take effect...[/cyan]")
            time.sleep(10)
            
            # Verify the fix
            console.print("\n[cyan]🔍 Verifying fix...[/cyan]")
            still_vulnerable = self.scan_for_vulnerability(vuln)
            
            attempt_record['verified'] = not still_vulnerable
            attempts.append(attempt_record)
            
            if not still_vulnerable:
                # SUCCESS!
                console.print("\n[bold green]🎉 VULNERABILITY FIXED! 🎉[/bold green]\n")
                
                # Track success pattern
                self.success_patterns.append({
                    'vuln_type': vuln.title,
                    'commands': remediation.proposed_commands,
                    'attempt': attempt_num
                })
                
                return {
                    'vuln_id': vuln.id,
                    'status': 'fixed',
                    'attempts': attempts,
                    'fixed_on_attempt': attempt_num
                }
            else:
                # Still vulnerable
                console.print("\n[yellow]⚠ Verification shows vulnerability still exists[/yellow]")
                
                if attempt_num < self.max_attempts:
                    console.print("[yellow]Will try a different approach...[/yellow]")
                    time.sleep(2)
        
        # All attempts exhausted
        console.print("\n[red]✗ Failed to fix after all attempts[/red]\n")
        
        # Track failure pattern
        self.failure_patterns.append({
            'vuln_type': vuln.title,
            'all_attempts': attempts
        })
        
        return {
            'vuln_id': vuln.id,
            'status': 'failed',
            'attempts': attempts,
            'fixed_on_attempt': None
        }

    def run_adaptive_loop(self, max_vulns: Optional[int] = None, min_severity: int = 2):
        """Run adaptive QA loop with feedback"""
        console.print(Panel.fit(
            "[bold cyan]Adaptive QA Agent[/bold cyan]\n"
            "Self-correcting with feedback loops",
            border_style="cyan"
        ))
        
        # Initial scan
        console.print("\n[bold cyan]Running Initial Scan...[/bold cyan]\n")
        scan_file = self.work_dir / "initial_scan.xml"
        parsed_file = self.work_dir / "initial_scan_parsed.json"
        
        success = self.scanner.run_scan(
            profile=self.scan_profile,
            output_file="/tmp/initial_scan.xml",
            datastream=self.scan_datastream,
            sudo_password=self.sudo_password
        )
        
        if not success:
            console.print("[red]Initial scan failed![/red]")
            sys.exit(1)
        
        self.scanner.download_results("/tmp/initial_scan.xml", str(scan_file))
        parse_openscap(str(scan_file), str(parsed_file))
        
        # Load vulnerabilities
        with open(parsed_file) as f:
            vulns_data = json.load(f)
        
        vulns = [Vulnerability(**v) for v in vulns_data]
        
        # Filter
        filtered = [v for v in vulns if int(v.severity) >= min_severity]
        console.print(f"\n[yellow]Found {len(filtered)} vulnerabilities (severity >= {min_severity})[/yellow]")
        
        if max_vulns and len(filtered) > max_vulns:
            filtered = filtered[:max_vulns]
            console.print(f"[yellow]Limiting to first {max_vulns} vulnerabilities[/yellow]\n")
        
        # Process each vulnerability
        all_results = []
        
        for i, vuln in enumerate(filtered, 1):
            console.print(f"\n[bold cyan]╔═══ Vulnerability {i}/{len(filtered)} ═══╗[/bold cyan]")
            
            result = self.process_vulnerability_adaptively(vuln)
            all_results.append(result)
            
            # Update tracking
            if result['status'] == 'fixed':
                if result['fixed_on_attempt'] == 1:
                    self.results['fixed_first_try'].append(result['vuln_id'])
                else:
                    self.results['fixed_after_retry'].append(result['vuln_id'])
            else:
                self.results['failed_all_attempts'].append(result['vuln_id'])
            
            # Save intermediate results
            self._save_results(all_results)
            
            # Show progress
            self._show_progress()
            
            # Continue?
            if i < len(filtered):
                if not Confirm.ask("\n[bold]Continue to next vulnerability?[/bold]", default=True):
                    break
        
        # Final summary
        self._show_final_summary(all_results)
        # Write text report (always)
        try:
            self._write_text_report(all_results)
            console.print(f"\n[green]Text report saved: {self.work_dir}/adaptive_report.txt[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to write text report: {e}[/yellow]")
        # Write PDF report (optional)
        try:
            self._write_pdf_report(all_results)
            console.print(f"\n[green]PDF report saved: {self.work_dir}/adaptive_report.pdf[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to write PDF report: {e}[/yellow]")

    def _show_progress(self):
        """Show current progress"""
        total = (len(self.results['fixed_first_try']) + 
                len(self.results['fixed_after_retry']) + 
                len(self.results['failed_all_attempts']))
        
        console.print(f"\n[dim]Progress: {total} processed[/dim]")
        console.print(f"[dim]  ✓ Fixed (1st try): {len(self.results['fixed_first_try'])}[/dim]")
        console.print(f"[dim]  ✓ Fixed (retry): {len(self.results['fixed_after_retry'])}[/dim]")
        console.print(f"[dim]  ✗ Failed: {len(self.results['failed_all_attempts'])}[/dim]")
    
    def _save_results(self, all_results: List[Dict]):
        """Save intermediate results"""
        results_file = self.work_dir / "adaptive_results.json"
        results_file.write_text(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'summary': self.results,
            'detailed_results': all_results,
            'success_patterns': self.success_patterns,
            'failure_patterns': self.failure_patterns
        }, indent=2))
    
    def _show_final_summary(self, all_results: List[Dict]):
        """Show final summary"""
        console.print("\n" + "="*70)
        console.print("[bold cyan]Adaptive QA Agent - Final Summary[/bold cyan]")
        console.print("="*70 + "\n")
        
        table = Table(title="Results")
        table.add_column("Outcome", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        table.add_column("Details", style="dim")
        
        table.add_row(
            "✓ Fixed (First Try)",
            str(len(self.results['fixed_first_try'])),
            "No retries needed",
            style="green"
        )
        table.add_row(
            "✓ Fixed (After Retry)",
            str(len(self.results['fixed_after_retry'])),
            "Agent adapted strategy",
            style="green"
        )
        table.add_row(
            "✗ Failed",
            str(len(self.results['failed_all_attempts'])),
            f"All {self.max_attempts} attempts failed",
            style="red"
        )
        
        console.print(table)
        
        # Show learning insights
        if self.success_patterns:
            console.print("\n[bold green]🎓 Learning: Successful Patterns[/bold green]")
            for pattern in self.success_patterns[:3]:
                console.print(f"  • {pattern['vuln_type']}: Fixed on attempt {pattern['attempt']}")
        
        # Show detailed results
        console.print("\n[bold]Detailed Results:[/bold]\n")
        for result in all_results:
            status_icon = "✓" if result['status'] == 'fixed' else "✗"
            status_color = "green" if result['status'] == 'fixed' else "red"
            
            console.print(f"[{status_color}]{status_icon} {result['vuln_id']}[/{status_color}]")
            console.print(f"   Attempts: {len(result['attempts'])}")
            if result['status'] == 'fixed':
                console.print(f"   Fixed on: Attempt {result['fixed_on_attempt']}")
        
        console.print(f"\n[green]Results saved: {self.work_dir}/adaptive_results.json[/green]")

    def _write_text_report(self, all_results: List[Dict]):
        """Generate a plain-text report summarizing attempts, playbooks, and outputs."""
        report_path = self.work_dir / "adaptive_report.txt"
        lines: List[str] = []
        lines.append("Adaptive QA Agent Report\n")
        lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        try:
            host = getattr(self.scanner, 'target_host', 'unknown-host')
        except Exception:
            host = 'unknown-host'
        lines.append(f"Target Host: {host}\n")
        lines.append("="*80 + "\n\n")

        for result in all_results:
            vuln_id = result.get('vuln_id', 'unknown')
            status = result.get('status', 'unknown')
            fixed_on = result.get('fixed_on_attempt')
            attempts = result.get('attempts', [])

            lines.append(f"Vulnerability: {vuln_id}\n")
            lines.append(f"Status: {status} | Fixed on attempt: {fixed_on if fixed_on else '-'}\n")
            lines.append("-"*80 + "\n")

            for att in attempts:
                attempt_num = att.get('attempt')
                cmds = att.get('commands', [])
                apply_success = att.get('apply_success')
                verified = att.get('verified')

                lines.append(f"Attempt {attempt_num}\n")
                lines.append("Commands:\n")
                for cmd in cmds:
                    lines.append(f"  - {cmd}\n")

                # Include commands file (or playbook if present for legacy runs)
                cmds_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.cmds.txt"
                playbook_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.yml"
                if cmds_path.exists():
                    lines.append(f"Commands File: {cmds_path.name}\n")
                    try:
                        cmds_text = cmds_path.read_text()
                    except Exception:
                        cmds_text = "<unable to read>"
                    lines.append("Commands Content:\n")
                    lines.append(cmds_text + ("\n" if not cmds_text.endswith("\n") else ""))
                elif playbook_path.exists():
                    lines.append(f"Playbook: {playbook_path.name}\n")
                    try:
                        playbook_text = playbook_path.read_text()
                    except Exception:
                        playbook_text = "<unable to read>"
                    lines.append("Playbook Content:\n")
                    lines.append(playbook_text + ("\n" if not playbook_text.endswith("\n") else ""))

                # Include ansible/ssh output
                log_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.ssh.log"
                if not log_path.exists():
                    # Fallback to ansible log if this run used Ansible
                    log_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.log"
                lines.append(f"Output (Apply: {'SUCCESS' if apply_success else 'FAILED'} | Verify: {'FIXED' if verified else 'PERSISTING' if verified is not None else 'N/A'}):\n")
                try:
                    log_text = log_path.read_text() if log_path.exists() else "<missing>"
                except Exception:
                    log_text = "<unable to read>"
                lines.append(log_text + ("\n" if not log_text.endswith("\n") else ""))

                lines.append("-"*80 + "\n")
            lines.append("\n")

        # Write the file
        report_path.write_text("".join(lines))

    def _write_pdf_report(self, all_results: List[Dict]):
        """Generate a PDF report summarizing attempts, playbooks, and outputs."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import simpleSplit
        except Exception as e:
            # Surface a clear error to the caller; caller prints a warning and continues
            raise RuntimeError("reportlab is not installed. Install with 'pip install reportlab' or 'pip install -r requirements.txt'") from e

        pdf_path = self.work_dir / "adaptive_report.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        def draw_wrapped(text: str, x: int, y: int, max_width: int, line_height: int = 14):
            lines = simpleSplit(text or "", "Helvetica", 10, max_width)
            cur_y = y
            for line in lines:
                if cur_y < 50:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    cur_y = height - 50
                c.drawString(x, cur_y, line)
                cur_y -= line_height
            return cur_y

        # Title
        c.setTitle("Adaptive QA Agent Report")
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Adaptive QA Agent Report")
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
        # Host
        try:
            host = getattr(self.scanner, 'target_host', 'unknown-host')
        except Exception:
            host = 'unknown-host'
        c.drawString(50, height - 85, f"Target Host: {host}")
        c.showPage()

        for result in all_results:
            vuln_id = result.get('vuln_id', 'unknown')
            status = result.get('status', 'unknown')
            fixed_on = result.get('fixed_on_attempt')
            attempts = result.get('attempts', [])

            # Page header for vulnerability
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 50, f"Vulnerability: {vuln_id}")
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 68, f"Status: {status}  |  Fixed on attempt: {fixed_on if fixed_on else '-'}")
            y = height - 90

            for att in attempts:
                attempt_num = att.get('attempt')
                cmds = att.get('commands', [])
                apply_success = att.get('apply_success')
                verified = att.get('verified')

                # Attempt header
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, f"Attempt {attempt_num}")
                y -= 18

                # Commands
                c.setFont("Helvetica-Bold", 10)
                c.drawString(50, y, "Commands:")
                y -= 14
                c.setFont("Helvetica", 10)
                y = draw_wrapped("\n".join(f"- {cmd}" for cmd in cmds), 60, y, width - 90)
                y -= 6

                # Playbook content
                playbook_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.yml"
                playbook_text = ""
                if playbook_path.exists():
                    try:
                        playbook_text = playbook_path.read_text()
                    except Exception:
                        playbook_text = "<unable to read playbook>"
                c.setFont("Helvetica-Bold", 10)
                c.drawString(50, y, f"Playbook: {playbook_path.name}")
                y -= 14
                c.setFont("Helvetica", 10)
                y = draw_wrapped(playbook_text[:4000], 60, y, width - 90)
                y -= 6

                # Ansible output
                log_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.log"
                log_text = ""
                if log_path.exists():
                    try:
                        log_text = log_path.read_text()
                    except Exception:
                        log_text = "<unable to read log>"
                c.setFont("Helvetica-Bold", 10)
                status_str = f"Apply: {'SUCCESS' if apply_success else 'FAILED'}  |  Verify: {'FIXED' if verified else 'PERSISTING' if verified is not None else 'N/A'}"
                c.drawString(50, y, f"Output ({status_str}):")
                y -= 14
                c.setFont("Helvetica", 10)
                y = draw_wrapped(log_text[:8000], 60, y, width - 90)
                y -= 12

                if y < 120:
                    c.showPage()
                    y = height - 50

            c.showPage()

        c.save()


class SSHExecutor:
    """Execute commands remotely over SSH and capture stdout/stderr."""
    def __init__(self, host: str, user: str, key: Optional[str], port: int, sudo_password: Optional[str]):
        self.host = host
        self.user = user
        self.key = key
        self.port = port
        self.sudo_password = sudo_password

    def _build_base_cmd(self) -> List[str]:
        cmd = ["ssh"]
        if self.key:
            cmd.extend(["-i", self.key])
        cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-p", str(self.port),
            f"{self.user}@{self.host}",
        ])
        return cmd

    def execute_commands(self, commands: List[str]) -> Tuple[bool, str]:
        all_ok = True
        logs: List[str] = []
        for idx, command in enumerate(commands, start=1):
            remote = f"bash -lc {shlex.quote(command)}"
            ssh_cmd = self._build_base_cmd() + [remote]
            logs.append(f"--- Command {idx} ---\n$ {command}\n")
            try:
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=600)
                if result.stdout:
                    logs.append(result.stdout)
                if result.stderr:
                    logs.append(result.stderr)
                if result.returncode != 0:
                    all_ok = False
                    logs.append(f"[exit_code={result.returncode}]\n")
            except subprocess.TimeoutExpired:
                all_ok = False
                logs.append("[timeout] Command timed out (>600s)\n")
            except FileNotFoundError:
                all_ok = False
                logs.append("[error] ssh not found on local system\n")
            except Exception as e:
                all_ok = False
                logs.append(f"[error] {e}\n")
            logs.append("\n")
        return all_ok, "".join(logs)
    
    def process_vulnerability_adaptively(self, vuln: Vulnerability) -> Dict:
        """Process a vulnerability with adaptive retries
        
        Returns dict with results of all attempts
        """
        console.print("\n" + "="*70)
        console.print(f"[bold cyan]Processing: {vuln.title}[/bold cyan]")
        console.print("="*70 + "\n")
        
        # Show vulnerability details
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("ID", vuln.id)
        table.add_row("Title", vuln.title)
        table.add_row("Severity", f"[{'red' if int(vuln.severity) >= 3 else 'yellow'}]{vuln.severity}[/]")
        table.add_row("Host", vuln.host)
        console.print(table)
        
        attempts = []
        
        for attempt_num in range(1, self.max_attempts + 1):
            console.print(f"\n[bold yellow]═══ Attempt {attempt_num}/{self.max_attempts} ═══[/bold yellow]\n")
            
            # Get remediation suggestion
            if attempt_num == 1:
                # First attempt: get initial suggestion
                remediation = self.get_initial_remediation(vuln)
            else:
                # Subsequent attempts: adaptive based on previous failures
                last_error = attempts[-1].get('error', 'Unknown error')
                remediation = self.get_adaptive_remediation(vuln, attempts, last_error)
            
            # Show remediation
            console.print("\n[green]💡 Remediation Plan:[/green]")
            for i, cmd in enumerate(remediation.proposed_commands, 1):
                console.print(f"  {i}. [yellow]{cmd}[/yellow]")
            if remediation.notes:
                console.print(f"\n[dim]Notes: {remediation.notes}[/dim]")
            
            # Apply remediation
            time.sleep(1)  # Brief pause for readability
            apply_success, output = self.apply_remediation(vuln, remediation, attempt_num)
            
            # Record attempt
            attempt_record = {
                'attempt': attempt_num,
                'commands': remediation.proposed_commands,
                'apply_success': apply_success,
                'error': output if not apply_success else None
            }
            
            if not apply_success:
                console.print("[red]✗ Playbook execution failed[/red]")
                attempts.append(attempt_record)
                
                # Show error and ask if should continue
                if attempt_num < self.max_attempts:
                    console.print("\n[yellow]Will retry with different approach...[/yellow]")
                    time.sleep(2)
                continue
            
            console.print("[green]✓ Playbook executed successfully[/green]")
            
            # Wait for changes to take effect
            console.print("\n[cyan]⏳ Waiting 10 seconds for changes to take effect...[/cyan]")
            time.sleep(10)
            
            # Verify the fix
            console.print("\n[cyan]🔍 Verifying fix...[/cyan]")
            still_vulnerable = self.scan_for_vulnerability(vuln)
            
            attempt_record['verified'] = not still_vulnerable
            attempts.append(attempt_record)
            
            if not still_vulnerable:
                # SUCCESS!
                console.print("\n[bold green]🎉 VULNERABILITY FIXED! 🎉[/bold green]\n")
                
                # Track success pattern
                self.success_patterns.append({
                    'vuln_type': vuln.title,
                    'commands': remediation.proposed_commands,
                    'attempt': attempt_num
                })
                
                return {
                    'vuln_id': vuln.id,
                    'status': 'fixed',
                    'attempts': attempts,
                    'fixed_on_attempt': attempt_num
                }
            else:
                # Still vulnerable
                console.print("\n[yellow]⚠ Verification shows vulnerability still exists[/yellow]")
                
                if attempt_num < self.max_attempts:
                    console.print("[yellow]Will try a different approach...[/yellow]")
                    time.sleep(2)
        
        # All attempts exhausted
        console.print("\n[red]✗ Failed to fix after all attempts[/red]\n")
        
        # Track failure pattern
        self.failure_patterns.append({
            'vuln_type': vuln.title,
            'all_attempts': attempts
        })
        
        return {
            'vuln_id': vuln.id,
            'status': 'failed',
            'attempts': attempts,
            'fixed_on_attempt': None
        }
    
    def run_adaptive_loop(self, max_vulns: Optional[int] = None, min_severity: int = 2):
        """Run adaptive QA loop with feedback"""
        console.print(Panel.fit(
            "[bold cyan]Adaptive QA Agent[/bold cyan]\n"
            "Self-correcting with feedback loops",
            border_style="cyan"
        ))
        
        # Initial scan
        console.print("\n[bold cyan]Running Initial Scan...[/bold cyan]\n")
        scan_file = self.work_dir / "initial_scan.xml"
        parsed_file = self.work_dir / "initial_scan_parsed.json"
        
        success = self.scanner.run_scan(
            profile=self.scan_profile,
            output_file="/tmp/initial_scan.xml",
            datastream=self.scan_datastream,
            sudo_password=self.sudo_password
        )
        
        if not success:
            console.print("[red]Initial scan failed![/red]")
            sys.exit(1)
        
        self.scanner.download_results("/tmp/initial_scan.xml", str(scan_file))
        parse_openscap(str(scan_file), str(parsed_file))
        
        # Load vulnerabilities
        with open(parsed_file) as f:
            vulns_data = json.load(f)
        
        vulns = [Vulnerability(**v) for v in vulns_data]
        
        # Filter
        filtered = [v for v in vulns if int(v.severity) >= min_severity]
        console.print(f"\n[yellow]Found {len(filtered)} vulnerabilities (severity >= {min_severity})[/yellow]")
        
        if max_vulns and len(filtered) > max_vulns:
            filtered = filtered[:max_vulns]
            console.print(f"[yellow]Limiting to first {max_vulns} vulnerabilities[/yellow]\n")
        
        # Process each vulnerability
        all_results = []
        
        for i, vuln in enumerate(filtered, 1):
            console.print(f"\n[bold cyan]╔═══ Vulnerability {i}/{len(filtered)} ═══╗[/bold cyan]")
            
            result = self.process_vulnerability_adaptively(vuln)
            all_results.append(result)
            
            # Update tracking
            if result['status'] == 'fixed':
                if result['fixed_on_attempt'] == 1:
                    self.results['fixed_first_try'].append(result['vuln_id'])
                else:
                    self.results['fixed_after_retry'].append(result['vuln_id'])
            else:
                self.results['failed_all_attempts'].append(result['vuln_id'])
            
            # Save intermediate results
            self._save_results(all_results)
            
            # Show progress
            self._show_progress()
            
            # Continue?
            if i < len(filtered):
                if not Confirm.ask("\n[bold]Continue to next vulnerability?[/bold]", default=True):
                    break
        
        # Final summary
        self._show_final_summary(all_results)
        # Write text report (always)
        try:
            self._write_text_report(all_results)
            console.print(f"\n[green]Text report saved: {self.work_dir}/adaptive_report.txt[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to write text report: {e}[/yellow]")
        # Write PDF report (optional)
        try:
            self._write_pdf_report(all_results)
            console.print(f"\n[green]PDF report saved: {self.work_dir}/adaptive_report.pdf[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to write PDF report: {e}[/yellow]")
    
    def _show_progress(self):
        """Show current progress"""
        total = (len(self.results['fixed_first_try']) + 
                len(self.results['fixed_after_retry']) + 
                len(self.results['failed_all_attempts']))
        
        console.print(f"\n[dim]Progress: {total} processed[/dim]")
        console.print(f"[dim]  ✓ Fixed (1st try): {len(self.results['fixed_first_try'])}[/dim]")
        console.print(f"[dim]  ✓ Fixed (retry): {len(self.results['fixed_after_retry'])}[/dim]")
        console.print(f"[dim]  ✗ Failed: {len(self.results['failed_all_attempts'])}[/dim]")
    
    def _save_results(self, all_results: List[Dict]):
        """Save intermediate results"""
        results_file = self.work_dir / "adaptive_results.json"
        results_file.write_text(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'summary': self.results,
            'detailed_results': all_results,
            'success_patterns': self.success_patterns,
            'failure_patterns': self.failure_patterns
        }, indent=2))
    
    def _show_final_summary(self, all_results: List[Dict]):
        """Show final summary"""
        console.print("\n" + "="*70)
        console.print("[bold cyan]Adaptive QA Agent - Final Summary[/bold cyan]")
        console.print("="*70 + "\n")
        
        table = Table(title="Results")
        table.add_column("Outcome", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        table.add_column("Details", style="dim")
        
        table.add_row(
            "✓ Fixed (First Try)",
            str(len(self.results['fixed_first_try'])),
            "No retries needed",
            style="green"
        )
        table.add_row(
            "✓ Fixed (After Retry)",
            str(len(self.results['fixed_after_retry'])),
            "Agent adapted strategy",
            style="green"
        )
        table.add_row(
            "✗ Failed",
            str(len(self.results['failed_all_attempts'])),
            f"All {self.max_attempts} attempts failed",
            style="red"
        )
        
        console.print(table)
        
        # Show learning insights
        if self.success_patterns:
            console.print("\n[bold green]🎓 Learning: Successful Patterns[/bold green]")
            for pattern in self.success_patterns[:3]:
                console.print(f"  • {pattern['vuln_type']}: Fixed on attempt {pattern['attempt']}")
        
        # Show detailed results
        console.print("\n[bold]Detailed Results:[/bold]\n")
        for result in all_results:
            status_icon = "✓" if result['status'] == 'fixed' else "✗"
            status_color = "green" if result['status'] == 'fixed' else "red"
            
            console.print(f"[{status_color}]{status_icon} {result['vuln_id']}[/{status_color}]")
            console.print(f"   Attempts: {len(result['attempts'])}")
            if result['status'] == 'fixed':
                console.print(f"   Fixed on: Attempt {result['fixed_on_attempt']}")
        
        console.print(f"\n[green]Results saved: {self.work_dir}/adaptive_results.json[/green]")

    def _write_text_report(self, all_results: List[Dict]):
        """Generate a plain-text report summarizing attempts, playbooks, and outputs."""
        report_path = self.work_dir / "adaptive_report.txt"
        lines: List[str] = []
        lines.append("Adaptive QA Agent Report\n")
        lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        try:
            host = getattr(self.scanner, 'target_host', 'unknown-host')
        except Exception:
            host = 'unknown-host'
        lines.append(f"Target Host: {host}\n")
        lines.append("="*80 + "\n\n")

        for result in all_results:
            vuln_id = result.get('vuln_id', 'unknown')
            status = result.get('status', 'unknown')
            fixed_on = result.get('fixed_on_attempt')
            attempts = result.get('attempts', [])

            lines.append(f"Vulnerability: {vuln_id}\n")
            lines.append(f"Status: {status} | Fixed on attempt: {fixed_on if fixed_on else '-'}\n")
            lines.append("-"*80 + "\n")

            for att in attempts:
                attempt_num = att.get('attempt')
                cmds = att.get('commands', [])
                apply_success = att.get('apply_success')
                verified = att.get('verified')

                lines.append(f"Attempt {attempt_num}\n")
                lines.append("Commands:\n")
                for cmd in cmds:
                    lines.append(f"  - {cmd}\n")

                # Include commands file (or playbook if present for legacy runs)
                cmds_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.cmds.txt"
                playbook_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.yml"
                if cmds_path.exists():
                    lines.append(f"Commands File: {cmds_path.name}\n")
                    try:
                        cmds_text = cmds_path.read_text()
                    except Exception:
                        cmds_text = "<unable to read>"
                    lines.append("Commands Content:\n")
                    lines.append(cmds_text + ("\n" if not cmds_text.endswith("\n") else ""))
                elif playbook_path.exists():
                    lines.append(f"Playbook: {playbook_path.name}\n")
                    try:
                        playbook_text = playbook_path.read_text()
                    except Exception:
                        playbook_text = "<unable to read>"
                    lines.append("Playbook Content:\n")
                    lines.append(playbook_text + ("\n" if not playbook_text.endswith("\n") else ""))

                # Include ansible output
                log_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.ssh.log"
                if not log_path.exists():
                    # Fallback to ansible log if this run used Ansible
                    log_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.log"
                lines.append(f"Output (Apply: {'SUCCESS' if apply_success else 'FAILED'} | Verify: {'FIXED' if verified else 'PERSISTING' if verified is not None else 'N/A'}):\n")
                try:
                    log_text = log_path.read_text() if log_path.exists() else "<missing>"
                except Exception:
                    log_text = "<unable to read>"
                lines.append(log_text + ("\n" if not log_text.endswith("\n") else ""))

                lines.append("-"*80 + "\n")
            lines.append("\n")

        # Write the file
        report_path.write_text("".join(lines))

    def _write_pdf_report(self, all_results: List[Dict]):
        """Generate a PDF report summarizing attempts, playbooks, and outputs."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import simpleSplit
        except Exception as e:
            # Surface a clear error to the caller; caller prints a warning and continues
            raise RuntimeError("reportlab is not installed. Install with 'pip install reportlab' or 'pip install -r requirements.txt'") from e

        pdf_path = self.work_dir / "adaptive_report.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        def draw_wrapped(text: str, x: int, y: int, max_width: int, line_height: int = 14):
            lines = simpleSplit(text or "", "Helvetica", 10, max_width)
            cur_y = y
            for line in lines:
                if cur_y < 50:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    cur_y = height - 50
                c.drawString(x, cur_y, line)
                cur_y -= line_height
            return cur_y

        # Title
        c.setTitle("Adaptive QA Agent Report")
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Adaptive QA Agent Report")
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
        # Host
        try:
            host = getattr(self.scanner, 'target_host', 'unknown-host')
        except Exception:
            host = 'unknown-host'
        c.drawString(50, height - 85, f"Target Host: {host}")
        c.showPage()

        for result in all_results:
            vuln_id = result.get('vuln_id', 'unknown')
            status = result.get('status', 'unknown')
            fixed_on = result.get('fixed_on_attempt')
            attempts = result.get('attempts', [])

            # Page header for vulnerability
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 50, f"Vulnerability: {vuln_id}")
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 68, f"Status: {status}  |  Fixed on attempt: {fixed_on if fixed_on else '-'}")
            y = height - 90

            for att in attempts:
                attempt_num = att.get('attempt')
                cmds = att.get('commands', [])
                apply_success = att.get('apply_success')
                verified = att.get('verified')

                # Attempt header
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, f"Attempt {attempt_num}")
                y -= 18

                # Commands
                c.setFont("Helvetica-Bold", 10)
                c.drawString(50, y, "Commands:")
                y -= 14
                c.setFont("Helvetica", 10)
                y = draw_wrapped("\n".join(f"- {cmd}" for cmd in cmds), 60, y, width - 90)
                y -= 6

                # Playbook content
                playbook_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.yml"
                playbook_text = ""
                if playbook_path.exists():
                    try:
                        playbook_text = playbook_path.read_text()
                    except Exception:
                        playbook_text = "<unable to read playbook>"
                c.setFont("Helvetica-Bold", 10)
                c.drawString(50, y, f"Playbook: {playbook_path.name}")
                y -= 14
                c.setFont("Helvetica", 10)
                y = draw_wrapped(playbook_text[:4000], 60, y, width - 90)
                y -= 6

                # Ansible output
                log_path = self.work_dir / f"fix_{vuln_id}_attempt{attempt_num}.log"
                log_text = ""
                if log_path.exists():
                    try:
                        log_text = log_path.read_text()
                    except Exception:
                        log_text = "<unable to read log>"
                c.setFont("Helvetica-Bold", 10)
                status_str = f"Apply: {'SUCCESS' if apply_success else 'FAILED'}  |  Verify: {'FIXED' if verified else 'PERSISTING' if verified is not None else 'N/A'}"
                c.drawString(50, y, f"Output ({status_str}):")
                y -= 14
                c.setFont("Helvetica", 10)
                y = draw_wrapped(log_text[:8000], 60, y, width - 90)
                y -= 12

                if y < 120:
                    c.showPage()
                    y = height - 50

            c.showPage()

        c.save()


def main():
    """Main entry point"""
    import argparse
    
    # Verify .env is loaded
    if not os.getenv('OPENROUTER_API_KEY'):
        console.print("[red]Error: OPENROUTER_API_KEY not found in .env file![/red]")
        console.print("[yellow]Please create .env from env.template and add your API key[/yellow]")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Adaptive QA Agent with feedback loops")
    parser.add_argument('--host', required=True)
    parser.add_argument('--user', default='root')
    parser.add_argument('--key', help='SSH key path')
    parser.add_argument('--sudo-password', help='Sudo password')
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--profile', default='xccdf_org.ssgproject.content_profile_cis')
    parser.add_argument('--datastream', default='/usr/share/xml/scap/ssg/content/ssg-rl10-ds.xml')
    parser.add_argument('--work-dir', default='adaptive_qa_work')
    parser.add_argument('--max-vulns', type=int, help='Max vulnerabilities to process')
    parser.add_argument('--min-severity', type=int, default=2, choices=[0,1,2,3,4])
    parser.add_argument('--max-attempts', type=int, default=5, help='Max retry attempts per vulnerability')
    
    args = parser.parse_args()
    
    # Create scanner
    scanner = OpenSCAPScanner(
        target_host=args.host,
        ssh_user=args.user,
        ssh_key=args.key,
        ssh_port=22
    )
    
    # Create adaptive agent
    agent = AdaptiveQAAgent(
        scanner=scanner,
        ansible_inventory=args.inventory,
        work_dir=Path(args.work_dir),
        scan_profile=args.profile,
        scan_datastream=args.datastream,
        sudo_password=args.sudo_password,
        max_attempts=args.max_attempts
    )
    
    try:
        # Call via class to avoid any instance attribute shadowing issues
        AdaptiveQAAgent.run_adaptive_loop(
            agent,
            max_vulns=args.max_vulns,
            min_severity=args.min_severity
        )
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

