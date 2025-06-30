import logging
import re

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'RESET': '\033[0m',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        formatted = super().format(record)
        
        # Use regex to find content inside square brackets and color only that content
        def color_brackets(match):
            return f"{color}{match.group(0)}{reset}"
        
        # Replace content inside square brackets with colored version
        colored_formatted = re.sub(r'\[[^\]]*\]', color_brackets, formatted)
        
        return colored_formatted 