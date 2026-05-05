# Retrieval Quality Eval — baseline-prod-with-llm (2026-05-04T21-12-18Z)

**Summary:** 4 / 8 passed

## [PASS] `biochar_finance_grounding` — 'biochar finance'
_Aaron's Issue #1 — definitional summary must come from YOE content, not training data (no biochar+asphalt riff)._

**Search method:** keyword_heavy • **Docs retrieved:** 5 • **Citations:** 2

**Cited:**
- EP 122: Episode 122 – Dr. David Laird, PhD, Biochar Science & Climate Stability (guest: Guest)
- EP 120: Episode 120 – Rowdy Yeatts, Founder & CEO, High Plains Biochar (guest: Guest)

**Response (first 800 chars):**

> In the context of biochar, finance can play a crucial role in supporting the development and implementation of biochar projects. Here are some key points to consider regarding biochar finance:
> 
> 1. **Investment Opportunities**: Biochar projects can offer investment opportunities for individuals, organizations, and impact investors looking to support sustainable agriculture, carbon sequestration, and soil health initiatives.
> 
> 2. **Grant Funding**: There are grants and funding opportunities available for biochar research, development, and implementation. Organizations and entrepreneurs in the biochar space can explore grant programs to support their projects.
> 
> 3. **Carbon Markets**: Biochar has the potential to sequester carbon in soils, which can create opportunities for generating carbon cr

## [FAIL] `finance_theme_saturation` — 'finance'
_Aaron's Issue #2 — short theme query must surface multiple IMPACT INVESTING-tagged episodes._

**Failures:**
- no cited episode tagged with category IMPACT INVESTING
- got 0 IMPACT INVESTING-tagged citations, need >= 3

**Search method:** hybrid • **Docs retrieved:** 5 • **Citations:** 3

**Cited:**
- BOOK: Y on Earth: Get Smarter, Feel Better, Heal the Planet — Ch. 20 (• Earn 267)
- EP 166: Episode 166 – Dahr Jamail, Communications & Media Relations Lead, Home Planet Fund (guest: Guest)
- EP 105: Episode 105 – NGO & Social Enterprise Accounting Intro with Aaron William Perry (guest: Guest)

**Response (first 800 chars):**

> Finance is a multifaceted field that encompasses the management of money, investments, and financial resources. In the context of sustainability and regenerative practices, finance plays a crucial role in supporting initiatives that prioritize environmental and social well-being. By aligning financial decisions with principles of sustainability, individuals and organizations can contribute to creating a more resilient and equitable economy.
> 
> Key aspects of finance to consider in the context of sustainability include:
> 
> 1. Impact Investing: Investing in companies, organizations, or projects that generate positive social or environmental impacts alongside financial returns.
> 
> 2. Green Finance: Financial products and services that support environmentally friendly projects and initiatives, such 

## [FAIL] `generative_technology_theme` — 'generative technology'
_Aaron's Issue #2 — must surface episodes tagged TECHNOLOGY & MATERIALS or REGEN / SOCIAL ENTERPRISE._

**Failures:**
- no cited episode is tagged with any of ['TECHNOLOGY & MATERIALS', 'REGEN / SOCIAL ENTERPRISE'] (cited episodes: [58, 79])

**Search method:** hybrid • **Docs retrieved:** 5 • **Citations:** 3

**Cited:**
- EP 58: Episode 58 – Sarah Drew, Visionary Author, Gaia Codex (guest: Guest)
- EP 79: Episode 79 – Brian Dillon, Growing Spaces & Rhythms of Thriving (guest: Guest)
- BOOK: Our Biggest Deal — Ch. 190 (| VIP GUEST ESSAYS)

**Response (first 800 chars):**

> Generative technology refers to technologies and innovations that have the capacity to create positive and regenerative impacts on the environment, society, and economy. Here are key aspects of generative technology:
> 
> 1. **Regenerative Capacity:** Generative technologies are designed to regenerate natural resources, enhance ecosystem health, and promote sustainable practices that benefit both humans and the environment.
> 
> 2. **Positive Externalities:** Generative technologies aim to generate positive externalities, such as social well-being, environmental stewardship, and economic prosperity, beyond their immediate intended outcomes.
> 
> 3. **Circular Economy Principles:** Generative technologies align with the principles of a circular economy, emphasizing resource efficiency, waste reduction,

## [FAIL] `soil_werks_regression` — 'Tell me about Soil Werks fertilizer'
_Regression — recently-ingested Soil Werks resource must still surface._

**Failures:**
- transport error: HTTP Error 504: Gateway Time-out
- response missing all of expected substrings ['Soil Werks', 'soilwerks']
- got 0 citations, need >= 1

**Search method:** ? • **Docs retrieved:** 0 • **Citations:** 0

**Cited:**

**Response (first 800 chars):**

> (empty)

## [PASS] `wele_waters_regression` — 'What does Wele Waters make?'
_Regression — recently-ingested Wele Waters resource must still surface._

**Search method:** hybrid • **Docs retrieved:** 5 • **Citations:** 3

**Cited:**
- BOOK: VIRIDITAS: THE GREAT HEALING — Ch. 33 (| Aaron William Perry)
- EP resource_welewaters: Wele Waters – Monthly Program (guest: Guest)
- EP 68: Episode 68 – Corona Virus Special w/ Aaron Perry & Artem Nikulkov (guest: Guest)

**Response (first 800 chars):**

> Wele Waters is a company that specializes in creating herbal medicine, aromatherapy products, and self-care items designed to promote health and well-being. Their products are crafted with a focus on holistic wellness and the healing properties of natural ingredients.
> 
> Wele Waters offers a range of products that include herbal medicine, aromatherapy blends, and self-care items to support individuals in their wellness journey. These products are made with high-quality, organic ingredients and are intended to provide nourishment for the body, mind, and spirit.
> 
> If you're interested in exploring the products offered by Wele Waters and learning more about their approach to herbal medicine and aromatherapy, I recommend visiting their website or listening to the specific podcast episodes where W

## [PASS] `viriditas_book_coverage` — 'What is VIRIDITAS about?'
_Aaron's Issue #3 — book by Aaron must surface as a book citation._

**Search method:** hybrid • **Docs retrieved:** 5 • **Citations:** 3

**Cited:**
- BOOK: VIRIDITAS: THE GREAT HEALING — Ch. 33 (| Aaron William Perry)
- BOOK: VIRIDITAS: THE GREAT HEALING — Ch. 30 (| Aaron William Perry)
- BOOK: VIRIDITAS: THE GREAT HEALING — Ch. 1 (| Aaron William Perry)

**Response (first 800 chars):**

> VIRIDITAS is a concept that encompasses the healing and life-force energy embodied by the plant kingdom. It represents the green life and vitality present in plants, as well as their inherent wisdom and capacity for healing. VIRIDITAS is understood as a potent force that can support human well-being and environmental restoration.
> 
> In the context of the YonEarth podcast episodes, VIRIDITAS is often discussed in relation to the Viriditas Society, a global network of Earth-tenders who collaborate with the plant kingdom for healing and wellness. Members of the Viriditas Society engage in practices that honor and work with the healing energies of plants, recognizing their role as ancient and sacred beings on Earth.
> 
> If you want to delve deeper into the concept of VIRIDITAS and its significance 

## [FAIL] `soil_stewardship_book_coverage` — 'How do I make biochar at home?'
_Aaron's Issue #3 — Soil Stewardship Handbook should surface for biochar how-to (it's a topic of the book)._

**Failures:**
- no cited book matches ['Soil Stewardship Handbook'] (cited books: [])

**Search method:** keyword_heavy • **Docs retrieved:** 5 • **Citations:** 2

**Cited:**
- EP 120: Episode 120 – Rowdy Yeatts, Founder & CEO, High Plains Biochar (guest: Guest)
- EP 122: Episode 122 – Dr. David Laird, PhD, Biochar Science & Climate Stability (guest: Guest)

**Response (first 800 chars):**

> Making biochar at home can be a straightforward process. Here's a basic guide to help you get started:
> 
> 1. **Materials Needed:**
>    - Organic waste material (such as wood chips, branches, or agricultural residues)
>    - Metal drum or a similar container with a lid
>    - Fire source (like a small burner or a fire pit)
> 
> 2. **Steps:**
>    - Fill the metal drum with organic waste material, leaving some space at the top.
>    - Place the drum on the fire source and light the material inside.
>    - Let the organic material burn until it turns into charcoal. Make sure to keep the lid on to control oxygen flow and prevent a full burn to ash.
>    - Once the material has turned into charcoal, carefully extinguish the fire and let the drum cool down.
> 
> 3. **Safety Precautions:**
>    - Always perform this proc

## [PASS] `y_on_earth_book_coverage` — 'What practices does the Y on Earth book recommend for daily life?'
_Aaron's Issue #3 — Y on Earth book by title must cite the book._

**Search method:** hybrid • **Docs retrieved:** 5 • **Citations:** 3

**Cited:**
- EP resource_soilwerks: Soil Werks (guest: Guest)
- BOOK: Soil Stewardship Handbook — Ch. 34 (AARON WILLIAM PERRY)
- BOOK: Y on Earth: Get Smarter, Feel Better, Heal the Planet — Ch. 112 (PART One • We Are What We Eat & We Are Soil)

**Response (first 800 chars):**

> The "Y on Earth" book offers practical guidance and recommendations for incorporating regenerative practices into daily life to promote well-being and sustainability. Some of the practices the book suggests include:
> 
> 1. **Mindfulness**: Cultivating awareness and presence in daily activities to enhance mental clarity, emotional balance, and connection to the world around us.
> 
> 2. **Healthy Eating**: Embracing plant-based diets, locally sourced foods, and mindful eating habits to support personal health and reduce environmental impact.
> 
> 3. **Physical Movement**: Engaging in regular physical exercise, outdoor activities, and mindful movement practices to promote physical health and connection to nature.
> 
> 4. **Connection to Soil**: Building a relationship with the soil through gardening, compos
