"""Turn a meeting transcript into action items that each link back to the exact lines they came from.

Pipeline: number the lines -> chunk -> ask Claude for structured items with a supporting quote
-> verify every quote in code -> write verified / review / rejected lists.
"""

__version__ = "0.1.0"
