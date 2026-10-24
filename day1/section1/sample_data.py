"""
===============================================================================
 Future with Shivank — AI Harness
 Section 1 · Sample input text shared by scripts 02, 03 and 04
===============================================================================

WHY THIS FILE EXISTS
--------------------
Every script in this section asks an LLM to summarise the SAME piece of text,
so you can compare a cloud model (OpenAI) with a local model (Ollama) on an
identical input. Keeping the text in one place means you only edit it once.

TRY THIS
--------
Replace PERSON_INFORMATION with a paragraph about anyone you like (a scientist,
a cricketer, your favourite founder) and re-run the scripts. Nothing else in
the code needs to change — that is the power of a prompt TEMPLATE.
===============================================================================
"""

# A short biography written for this course. Triple quotes let one Python
# string span many lines.
PERSON_INFORMATION = """
Avul Pakir Jainulabdeen Abdul Kalam (15 October 1931 – 27 July 2015) was an
Indian aerospace scientist and statesman who served as the 11th President of
India from 2002 to 2007. He was born into a modest family in Rameswaram, Tamil
Nadu, and as a boy he sold newspapers to support his family's income before
studying physics and then aerospace engineering at the Madras Institute of
Technology.

Kalam spent most of his scientific career at the Defence Research and
Development Organisation (DRDO) and the Indian Space Research Organisation
(ISRO). At ISRO he was project director of SLV-III, India's first indigenously
developed satellite launch vehicle, which placed the Rohini satellite into
orbit in 1980. He later led the development of several missile programmes,
which earned him the popular title "Missile Man of India", and he played a
central organisational role in the Pokhran-II nuclear tests in 1998.

He was awarded the Bharat Ratna, India's highest civilian honour, in 1997.
As President he became known as the "People's President" because of his
simple lifestyle and his habit of speaking directly with students across the
country. His autobiography, "Wings of Fire" (1999), became a bestseller and
remains popular with young readers. Kalam died on 27 July 2015 while
delivering a lecture to students at the Indian Institute of Management
Shillong, doing what he loved most: teaching.
"""

# ↑ Used like this in the other scripts:
#     from sample_data import PERSON_INFORMATION
#     chain.invoke({"information": PERSON_INFORMATION})
