"""
Comprehensive Alignment Screenshot Generator for all 12 corpus sources.
Features:
- Robust cookie banner & Sourcepoint CMP modal stripping (AU, taz, Hannover, etc.).
- Wayback Machine archive optimization (fast commit navigation, removes archive toolbar).
- Highlighting on BOTH Leichte Sprache (LS) and Standardsprache (AS) counterpart pages.
- Viewport captures, high-res close-ups, AS target captures, and paired side-by-side comparisons.
- Generates interactive HTML gallery and Markdown documentation report.
"""

import asyncio
import os
import sys
import argparse
import json
import traceback
from PIL import Image, ImageDraw, ImageFont

# Full configuration for all 12 corpus sources
SOURCE_CONFIGS = [
    {
        "id": "apotheken",
        "name": "Apotheken Umschau",
        "category": "Healthcare / Medical",
        "ls_url": "https://www.apotheken-umschau.de/einfache-sprache/krankheiten/weitsichtigkeit-753261.html",
        "as_url": "https://www.apotheken-umschau.de/krankheiten-symptome/augenkrankheiten/wie-kann-man-weitsichtigkeit-behandeln-733661.html",
        "strategy": "In-Text Cross-Reference Link",
        "description": "Am Ende des Leichte-Sprache-Artikels verlinkt die Apotheken Umschau mit Texten wie 'Hier finden Sie noch mehr Informationen über...' oder 'Informationen' auf den standardsprachlichen Fachartikel.",
        "ls_selector_js": """() => {
            const links = Array.from(document.querySelectorAll('a[href]'));
            const target = links.find(a => {
                const href = a.getAttribute('href') || '';
                const title = (a.getAttribute('title') || '').toLowerCase();
                const text = a.innerText.toLowerCase();
                return (!href.includes('/einfache-sprache/') && href.endsWith('.html') && (title.includes('hier') || title.includes('informationen') || text.includes('informationen') || text.includes('hier')));
            });
            return target ? (target.closest('p') || target) : null;
        }""",
        "ls_badge_text": "Alignment-Link: Referenz auf Standard-Fachartikel (AS)",
        "as_selector_js": """() => {
            return document.querySelector('h1.article-header__title') || 
                   document.querySelector('.article-header') || 
                   document.querySelector('h1') || 
                   document.querySelector('article');
        }""",
        "as_badge_text": "Standardsprachlicher Fachartikel (AS-Zielseite)"
    },
    {
        "id": "behindertenbeauftragter",
        "name": "Behindertenbeauftragter",
        "category": "Federal Government / Official",
        "ls_url": "https://www.behindertenbeauftragter.de/DE/LS/presse-und-aktuelles/veranstaltungen/sonderseiten/BRKKonferenz/FactSheet_07.html",
        "as_url": "https://www.behindertenbeauftragter.de/DE/AS/startseite/startseite-node.html",
        "strategy": "Header-Servicenavigation Sprachwechsler",
        "description": "In der oberen Metanavigation existiert ein Sprachumschalter (navServiceAS / Alltagssprache-Icon), der den direkten Wechsel zur Alltagssprache ermöglicht.",
        "ls_selector_js": """() => {
            return document.querySelector('li.navServiceAS') || 
                   document.querySelector('a[aria-label*=\"Alltagssprache\"]') || 
                   document.querySelector('.navServiceMeta') ||
                   document.querySelector('.c-language-switch');
        }""",
        "ls_badge_text": "Alignment-Wechsler: Umschaltung auf Alltagssprache (AS)",
        "ls_scroll_top": True,
        "as_selector_js": """() => {
            return document.querySelector('li.navServiceLS') || 
                   document.querySelector('a[aria-label*=\"Leichte Sprache\"]') || 
                   document.querySelector('.navServiceMeta') || 
                   document.querySelector('header');
        }""",
        "as_badge_text": "Umschalter zu Leichter Sprache (LS) aus AS-Sicht",
        "as_scroll_top": True
    },
    {
        "id": "brandeins",
        "name": "brand eins",
        "category": "Journalism / Economy (Archive)",
        "ls_url": "https://web.archive.org/web/20240528075943/https://www.brandeins.de/magazine/brand-eins-wirtschaftsmagazin/2022/abo-wirtschaft/leichte-sprache-hauptsache-es-stand-in-irgendeiner-liste",
        "as_url": None, # Dual in-page layout
        "strategy": "In-Page Parallele Textblöcke & Farbcodierung",
        "description": "brand eins stellt LS- und AS-Versionen direkt im selben Artikel gegenüber: Absätze in Alltagssprache (blau umrahmt) werden durch farbcodierte Absätze in Leichter Sprache (grün umrahmt) paarweise ergänzt.",
        "ls_selector_js": """() => {
            return document.querySelector('section.textblock') || document.querySelector('.article-body') || document.querySelector('article');
        }""",
        "custom_highlight_js": """(target) => {
            if (!target) return;
            target.scrollIntoView({ behavior: 'instant', block: 'center' });
            const paragraphs = Array.from(target.querySelectorAll('p'));
            if (paragraphs.length >= 1) {
                // AS Paragraph
                const pAS = paragraphs[0];
                pAS.style.outline = '4px solid #3b82f6';
                pAS.style.boxShadow = '0 0 20px rgba(59, 130, 246, 0.6)';
                pAS.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
                pAS.style.position = 'relative';
                pAS.style.borderRadius = '6px';
                pAS.style.padding = '8px';
                
                const bAS = document.createElement('div');
                bAS.innerText = 'Standardsprache (AS-Absatz)';
                bAS.style.position = 'absolute';
                bAS.style.top = '-30px';
                bAS.style.left = '0';
                bAS.style.background = '#3b82f6';
                bAS.style.color = '#fff';
                bAS.style.padding = '4px 10px';
                bAS.style.fontSize = '12px';
                bAS.style.fontWeight = 'bold';
                bAS.style.borderRadius = '4px';
                pAS.appendChild(bAS);
            }
            if (paragraphs.length >= 2) {
                // LS Paragraph
                const pLS = paragraphs[1];
                pLS.style.outline = '4px solid #10b981';
                pLS.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.6)';
                pLS.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
                pLS.style.position = 'relative';
                pLS.style.borderRadius = '6px';
                pLS.style.padding = '8px';
                pLS.style.marginTop = '35px';
                
                const bLS = document.createElement('div');
                bLS.innerText = 'Leichte Sprache (LS-Absatz)';
                bLS.style.position = 'absolute';
                bLS.style.top = '-30px';
                bLS.style.left = '0';
                bLS.style.background = '#10b981';
                bLS.style.color = '#fff';
                bLS.style.padding = '4px 10px';
                bLS.style.fontSize = '12px';
                bLS.style.fontWeight = 'bold';
                bLS.style.borderRadius = '4px';
                pLS.appendChild(bLS);
            }
        }""",
        "ls_badge_text": "Parallele Struktur: Blau = AS-Absatz, Grün = LS-Absatz"
    },
    {
        "id": "hamburg",
        "name": "Hamburg.de",
        "category": "Municipality / Portal",
        "ls_url": "https://www.hamburg.de/barrierefrei/leichte-sprache/polizei-feuerwehr/ls-starkregen-1019916",
        "as_url": "https://www.hamburg.de/politik-und-verwaltung/behoerden/bukea/themen/klima/starkregen-946792",
        "strategy": "Dedizierte Language-Bar ('Originaltext')",
        "description": "Jedem Leichte-Sprache-Artikel ist eine strukturierte Sprachleiste (.km1-language-bar) mit dem Button 'Originaltext / Alltagssprache' vorangestellt.",
        "ls_selector_js": """() => {
            return document.querySelector('.km1-language-bar') || 
                   document.querySelector('.km1-language-bar__btn-wrapper') || 
                   document.querySelector('a.original-text');
        }""",
        "ls_badge_text": "Alignment-Bar: 'Originaltext'-Button zum Fachartikel",
        "as_selector_js": """() => {
            return document.querySelector('.km1-language-bar') || 
                   document.querySelector('h1.km1-heading') || 
                   document.querySelector('h1') || 
                   document.querySelector('.km1-article-header');
        }""",
        "as_badge_text": "Standardsprachlicher Fachartikel (AS-Gegenstück)"
    },
    {
        "id": "hannover",
        "name": "Hannover.de",
        "category": "Municipality / Portal",
        "ls_url": "https://www.hannover.de/Leichte-Sprache/Hannover-und-Region/Politik/Wahlen/Kommunal∙wahlen-2026-in-der-Region-Hannover",
        "as_url": "https://www.hannover.de/Leben-in-der-Region-Hannover/Politik/Wahlen-Statistik/Kommunalwahlen-2026-in-der-Region-Hannover",
        "strategy": "Button 'Zur Seite in Alltagssprache' & Canonical Link",
        "description": "Hannover.de platziert am Artikelanfang einen auffälligen Button 'Zur Seite in Alltagssprache' (.schwer.icon) und referenziert die AS-Version im Canonical Link.",
        "ls_selector_js": """() => {
            return document.querySelector('a.schwer') || 
                   document.querySelector('a.icon-schwere-sprache-dkl') || 
                   Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('alltagssprache'));
        }""",
        "ls_badge_text": "Alignment-Button: 'Zur Seite in Alltagssprache'",
        "as_selector_js": """() => {
            return document.querySelector('a.leicht') || 
                   document.querySelector('a.icon-leichte-sprache-dkl') || 
                   document.querySelector('h1.content-detail__title') || 
                   document.querySelector('h1') || 
                   document.querySelector('article');
        }""",
        "as_badge_text": "Standardsprachlicher Artikel (AS-Gegenstück)"
    },
    {
        "id": "koeln",
        "name": "Stadt Köln",
        "category": "Municipality / Portal (Archive)",
        "ls_url": "https://web.archive.org/web/20220804230818/https://www.stadt-koeln.de/leben-in-koeln/soziales/unterhalts-vorschuss",
        "as_url": "https://web.archive.org/web/20220401090142/https://www.stadt-koeln.de/service/produkt/unterhaltsvorschuss-1",
        "strategy": "In-Content Switch Link ('Alltags-Sprache lesen')",
        "description": "Im Hauptinhalt jedes LS-Dokuments führt der Navigationslink 'Alltags-Sprache lesen' direkt zur regulären Dienstleistungsseite.",
        "ls_selector_js": """() => {
            const link = Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('alltags-sprache lesen'));
            return link ? (link.closest('p') || link) : null;
        }""",
        "ls_badge_text": "Alignment-Link: 'Alltags-Sprache lesen'",
        "as_selector_js": """() => {
            return document.querySelector('h1#inhalt') || 
                   document.querySelector('h1') || 
                   document.querySelector('.infobox') || 
                   document.querySelector('main');
        }""",
        "as_badge_text": "Zugehörige Dienstleistungsseite (Standardsprache)"
    },
    {
        "id": "main_taunus",
        "name": "Lebenshilfe Main-Taunus",
        "category": "NGO / Inclusion (Archive)",
        "ls_url": "https://web.archive.org/web/20210122220843/https://www.lebenshilfe-main-taunus.de/ls/reisen.html",
        "as_url": "https://web.archive.org/web/20210225194538/https://www.lebenshilfe-main-taunus.de/reisen.html",
        "strategy": "Header-Sprachumschalter mit Tooltip/Icon",
        "description": "Ein verankerter Header-Button mit dem Attribut title='Auf Alltags-Sprache umstellen' und dem Text 'Alltags-Sprache' dient als Umschaltpunkt.",
        "ls_selector_js": """() => {
            return document.querySelector('a[title*=\"Alltags-Sprache\"]') || 
                   Array.from(document.querySelectorAll('a')).find(a => (a.getAttribute('title') || '').includes('Alltags-Sprache') || a.innerText.toLowerCase().includes('alltags-sprache'));
        }""",
        "ls_badge_text": "Alignment-Button: 'Auf Alltags-Sprache umstellen'",
        "as_selector_js": """() => {
            return document.querySelector('a[title*=\"Leichte-Sprache\"]') || 
                   Array.from(document.querySelectorAll('a')).find(a => (a.getAttribute('title') || '').includes('Leichte-Sprache') || a.innerText.toLowerCase().includes('leichte sprache')) ||
                   document.querySelector('header');
        }""",
        "as_badge_text": "Gegenüberstellung: Umschalter 'Auf Leichte-Sprache umstellen'"
    },
    {
        "id": "mdr",
        "name": "MDR (Mitteldeutscher Rundfunk)",
        "category": "Public Broadcasting / News",
        "ls_url": "https://www.mdr.de/nachrichten-leicht/leichte-sprache-sachsen-sachsenforst-waldbrand-saison-100.html",
        "as_url": "https://www.mdr.de/nachrichten/sachsen/wetter-fruehling-wochenende-waldbrand-gefahr-102.html",
        "strategy": "Teaser-Box 'In schwerer Sprache lesen'",
        "description": "Unterhalb des LS-Textes bindet der MDR eine Teaser-Box mit der Überschrift 'HIER KÖNNEN SIE DIESE NACHRICHT AUCH IN SCHWERER SPRACHE LESEN:' ein, die zum redaktionellen Fachartikel führt.",
        "ls_selector_js": """() => {
            const hl = Array.from(document.querySelectorAll('.conHeadline, h3, h4, p, div')).find(e => e.innerText.toLowerCase().includes('schwerer sprache lesen'));
            return hl ? (hl.closest('.box') || hl) : null;
        }""",
        "ls_badge_text": "Alignment-Box: 'In schwerer Sprache lesen' (MDR Teaser)",
        "as_selector_js": """() => {
            return document.querySelector('h1.article-header__title') || 
                   document.querySelector('.article-header') || 
                   document.querySelector('h1');
        }""",
        "as_badge_text": "Zugehöriger redaktioneller Nachrichtenartikel (AS)"
    },
    {
        "id": "sozialpolitik",
        "name": "Sozialpolitik.com",
        "category": "Educational / Federal Ministry (BMAS)",
        "ls_url": "https://www.sozialpolitik.com/es/auswirkungen-der-coronavirus-epidemie",
        "as_url": "https://www.sozialpolitik.com/auswirkungen-der-coronavirus-epidemie",
        "strategy": "Header Quick-Switch 'Standardsprache'",
        "description": "In der oberen Leiste ermöglicht der Button 'Standardsprache' (.underline.easy) das direkte Hin- und Herschalten zur Standardfassung der Bildungseinheit.",
        "ls_selector_js": """() => {
            return Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('standardsprache'));
        }""",
        "ls_badge_text": "Alignment-Schalter: 'Standardsprache'",
        "as_selector_js": """() => {
            return Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('leichte sprache') || a.innerText.toLowerCase().includes('einfache sprache')) || 
                   document.querySelector('h1') || 
                   document.querySelector('main');
        }""",
        "as_badge_text": "Alignment-Gegenstück: 'Einfache Sprache' Umschalter (AS-Sicht)"
    },
    {
        "id": "stuttgart",
        "name": "Stuttgart.de",
        "category": "Municipality / Portal",
        "ls_url": "https://www.stuttgart.de/leben/gesundheit/vorsorge/suchtpraevention?sp%3Aout=easy",
        "as_url": "https://www.stuttgart.de/leben/gesundheit/vorsorge/suchtpraevention",
        "strategy": "Content-Button 'Artikel in Alltags-Sprache' / URL-Parameter",
        "description": "Stuttgart.de verwendet einen Aktionsbutton 'Artikel in Alltags-Sprache' (.SP-Link) und steuert die Barrierefreiheit über den URL-Parameter ?sp:out=easy.",
        "ls_selector_js": """() => {
            return Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('artikel in alltags-sprache') || (a.getAttribute('href') || '').includes('sp:out='));
        }""",
        "ls_badge_text": "Alignment-Button: 'Artikel in Alltags-Sprache'",
        "as_selector_js": """() => {
            return Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('artikel in leichter sprache') || (a.getAttribute('href') || '').includes('out=easy')) ||
                   document.querySelector('h1') || 
                   document.querySelector('main');
        }""",
        "as_badge_text": "Alignment-Button: 'Artikel in Leichter Sprache' (AS-Sicht)"
    },
    {
        "id": "taz",
        "name": "taz (die tageszeitung)",
        "category": "Journalism / News",
        "ls_url": "https://taz.de/Leichte-Sprache/!5590875/",
        "as_url": "https://taz.de/Wahlzulassung-fuer-Betreute/!5588713/",
        "strategy": "Redaktioneller In-Text Quellverweis ('schwerer Text')",
        "description": "Die taz-Redaktion verlinkt im Text mit redaktionellen Hinweisen wie 'aus diesem „schweren“ Text' auf den originalen Hintergrundartikel.",
        "ls_selector_js": """() => {
            const link = Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('schweren') && (a.innerText.toLowerCase().includes('text') || a.innerText.toLowerCase().includes('artikel')));
            return link ? (link.closest('p') || link) : null;
        }""",
        "ls_badge_text": "Alignment-Quellverweis: Link zum Ausgangsartikel (AS)",
        "as_selector_js": """() => {
            return document.querySelector('h1.secthead, h1.article, h1') || 
                   document.querySelector('article') || 
                   document.querySelector('.article-header');
        }""",
        "as_badge_text": "Originaler redaktioneller Hintergrundartikel (AS)"
    },
    {
        "id": "wiesbaden",
        "name": "Wiesbaden.de",
        "category": "Municipality / Portal",
        "ls_url": "https://www.wiesbaden.de/vv/produkte/31/Fuehrerschein-umtauschen-online-beantragen?sp%3Aeasylanguage=1",
        "as_url": "https://www.wiesbaden.de/vv/produkte/31/Fuehrerschein-umtauschen-online-beantragen",
        "strategy": "Sprachleisten-Toggle 'Leichte Sprache' / 'Alltagssprache'",
        "description": "Wiesbaden.de besitzt in der Funktionsleiste einen Umschaltlink (.SP-Link--simple-language) mit Parameter ?sp:easylanguage=1, der direkt zwischen Normalfassung und barrierefreier Fassung wechselt.",
        "ls_selector_js": """() => {
            return document.querySelector('.SP-Link--simple-language') || 
                   document.querySelector('a[href*=\"easylanguage\"]') ||
                   Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('leichte sprache') || a.innerText.toLowerCase().includes('alltagssprache'));
        }""",
        "ls_badge_text": "Alignment-Toggle: 'Leichte Sprache' (Aktiviert)",
        "as_selector_js": """() => {
            return document.querySelector('.SP-Link--simple-language') || 
                   document.querySelector('a[href*=\"easylanguage\"]') || 
                   Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('leichte sprache'));
        }""",
        "as_badge_text": "Alignment-Toggle: 'Leichte Sprache' (In AS-Ansicht)"
    }
]


# Universal DOM cleaning & overlay removal script executed on every page load
UNIVERSAL_CLEANUP_JS = """() => {
    // 1. Remove Sourcepoint CMP modals & containers (AU, taz, Hannover)
    document.querySelectorAll('div[id^="sp_message_container"], iframe[id^="sp_message_iframe"], [class*="sp_message"]').forEach(e => e.remove());
    
    // 2. Remove Usercentrics, Klaro, Cookiebot, OneTrust & custom overlays
    document.querySelectorAll('#usercentrics-root, #cmpbox, #klaro, #CybotCookiebotDialog, .cookie-banner, .cookie-modal, .sp-cookie-consent, aside.modal, .cmp-container, #onetrust-consent-sdk').forEach(e => e.remove());
    
    // 3. Remove Wayback Machine archive toolbar & banners
    const wm = document.getElementById('wm-ipp-base') || document.getElementById('wm-ipp') || document.getElementById('don-extend');
    if (wm) wm.remove();
    
    // 4. Restore scrollability & normal overflow
    document.documentElement.style.setProperty('overflow', 'auto', 'important');
    document.body.style.setProperty('overflow', 'auto', 'important');
    document.documentElement.style.setProperty('position', 'static', 'important');
    document.body.style.setProperty('position', 'static', 'important');
}"""


def combine_side_by_side(img_ls_path, img_as_path, output_path):
    """Creates a paired side-by-side comparison image with crisp layout."""
    try:
        img_ls = Image.open(img_ls_path)
        img_as = Image.open(img_as_path)

        target_height = min(img_ls.height, img_as.height, 950)
        
        scale_ls = target_height / img_ls.height
        scale_as = target_height / img_as.height
        
        w_ls = int(img_ls.width * scale_ls)
        w_as = int(img_as.width * scale_as)
        
        resized_ls = img_ls.resize((w_ls, target_height), Image.Resampling.LANCZOS)
        resized_as = img_as.resize((w_as, target_height), Image.Resampling.LANCZOS)
        
        header_h = 44
        border_w = 4
        
        total_w = w_ls + w_as + border_w
        total_h = target_height + header_h
        
        combo = Image.new("RGB", (total_w, total_h), (24, 24, 27))
        draw = ImageDraw.Draw(combo)
        
        # Draw column labels
        draw.rectangle([(0, 0), (w_ls, header_h)], fill=(225, 29, 72))
        draw.rectangle([(w_ls + border_w, 0), (total_w, header_h)], fill=(37, 99, 235))
        
        # Paste images
        combo.paste(resized_ls, (0, header_h))
        combo.paste(resized_as, (w_ls + border_w, header_h))
        
        combo.save(output_path, quality=92)
        return True
    except Exception as e:
        print(f"Error combining side-by-side images: {e}")
        return False


async def load_and_clean_page(page, url, timeout=40000):
    """Navigates to URL and strips overlays and cookie modals reliably."""
    is_archive = "web.archive.org" in url
    try:
        # For Wayback Machine use commit / domcontentloaded, for live sites use domcontentloaded
        await page.goto(url, wait_until="commit" if is_archive else "domcontentloaded", timeout=timeout)
        if is_archive:
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=15000)
            except Exception:
                pass
        await page.wait_for_timeout(2000)
    except Exception as e:
        print(f"    Navigation notice for {url}: {e}")
        await page.wait_for_timeout(2500)
        
    # Apply universal cleanup twice to catch delayed cookie scripts
    await page.evaluate(UNIVERSAL_CLEANUP_JS)
    await page.wait_for_timeout(500)
    await page.evaluate(UNIVERSAL_CLEANUP_JS)


async def apply_annotation_and_highlight(page, selector_js, badge_text, color="#e11d48", scroll_top=False, custom_js=None):
    """Injects glowing borders and modern floating badges around target elements."""
    return await page.evaluate(f"""() => {{
        // Clean overlays again before highlighting
        ({UNIVERSAL_CLEANUP_JS})();
        
        if ({'true' if custom_js else 'false'}) {{
            const customFn = {custom_js if custom_js else 'null'};
            if (customFn) {{
                const targetFinder = {selector_js};
                const target = targetFinder();
                customFn(target);
                return true;
            }}
        }}
        
        const targetFinder = {selector_js};
        const target = targetFinder();
        if (!target) return false;
        
        if ({'true' if scroll_top else 'false'}) {{
            window.scrollTo(0, 0);
        }} else {{
            target.scrollIntoView({{ behavior: 'instant', block: 'center', inline: 'center' }});
        }}
        
        target.style.setProperty('outline', '4px solid {color}', 'important');
        target.style.setProperty('box-shadow', '0 0 25px {color}bf, 0 0 10px {color}80', 'important');
        target.style.setProperty('border-radius', '6px', 'important');
        target.style.setProperty('position', 'relative', 'important');
        target.style.setProperty('background-color', '{color}18', 'important');
        
        // Remove prior badges if any
        target.querySelectorAll('.alignment-annotation-badge').forEach(b => b.remove());
        
        // Create Badge
        const badge = document.createElement('div');
        badge.className = 'alignment-annotation-badge';
        badge.innerText = {json.dumps(badge_text)};
        badge.style.position = 'absolute';
        badge.style.top = '-32px';
        badge.style.left = '0';
        badge.style.backgroundColor = '{color}';
        badge.style.color = '#ffffff';
        badge.style.padding = '5px 12px';
        badge.style.fontSize = '13px';
        badge.style.fontWeight = 'bold';
        badge.style.fontFamily = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        badge.style.borderRadius = '5px';
        badge.style.zIndex = '99999999';
        badge.style.boxShadow = '0 4px 12px rgba(0,0,0,0.4)';
        badge.style.whiteSpace = 'nowrap';
        badge.style.pointerEvents = 'none';
        
        const arrow = document.createElement('div');
        arrow.style.position = 'absolute';
        arrow.style.bottom = '-6px';
        arrow.style.left = '16px';
        arrow.style.width = '0';
        arrow.style.height = '0';
        arrow.style.borderLeft = '6px solid transparent';
        arrow.style.borderRight = '6px solid transparent';
        arrow.style.borderTop = '6px solid {color}';
        badge.appendChild(arrow);
        
        target.appendChild(badge);
        return true;
    }}""")


async def capture_source(browser, cfg, output_dir, timeout=40000):
    source_id = cfg["id"]
    name = cfg["name"]
    ls_url = cfg["ls_url"]
    as_url = cfg.get("as_url")
    
    print(f"\n[{name}] Starting capture...")
    src_dir = os.path.join(output_dir, source_id)
    os.makedirs(src_dir, exist_ok=True)
    
    results = {
        "id": source_id,
        "name": name,
        "category": cfg.get("category", "General"),
        "strategy": cfg["strategy"],
        "description": cfg["description"],
        "ls_url": ls_url,
        "as_url": as_url,
        "files": {}
    }
    
    context = await browser.new_context(
        viewport={'width': 1366, 'height': 850},
        device_scale_factor=2,
        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    
    # 1. PROCESS LEICHTE SPRACHE (LS) PAGE
    page_ls = await context.new_page()
    try:
        print(f"  Navigating to LS URL: {ls_url}")
        await load_and_clean_page(page_ls, ls_url, timeout=timeout)
        
        # Highlight LS target element
        hl_ok = await apply_annotation_and_highlight(
            page_ls,
            selector_js=cfg["ls_selector_js"],
            badge_text=cfg["ls_badge_text"],
            color="#e11d48", # Vibrant Red
            scroll_top=cfg.get("ls_scroll_top", False),
            custom_js=cfg.get("custom_highlight_js")
        )
        await page_ls.wait_for_timeout(800)
        
        # Save LS Viewport Screenshot
        viewport_path = os.path.join(src_dir, f"{source_id}_viewport.png")
        await page_ls.screenshot(path=viewport_path, full_page=False)
        results["files"]["viewport"] = viewport_path
        print(f"  ✓ Saved LS Viewport Screenshot: {viewport_path}")
        
        # Save Element Close-up
        try:
            handle = await page_ls.evaluate_handle(cfg["ls_selector_js"])
            el = handle.as_element()
            if el:
                crop_path = os.path.join(src_dir, f"{source_id}_element_closeup.png")
                await el.screenshot(path=crop_path)
                results["files"]["closeup"] = crop_path
                print(f"  ✓ Saved Element Close-up: {crop_path}")
        except Exception as e:
            print(f"  Note: element closeup skipped: {e}")
            
    except Exception as e:
        print(f"  [FAIL] Error capturing LS page for {name}: {e}")
        traceback.print_exc()
    finally:
        await page_ls.close()
        
    # 2. PROCESS STANDARDS PRACHE (AS) COUNTERPART PAGE
    if as_url:
        page_as = await context.new_page()
        try:
            print(f"  Navigating to AS Target URL: {as_url}")
            await load_and_clean_page(page_as, as_url, timeout=timeout)
            
            # Highlight reciprocal AS element if selector provided
            if cfg.get("as_selector_js"):
                await apply_annotation_and_highlight(
                    page_as,
                    selector_js=cfg["as_selector_js"],
                    badge_text=cfg.get("as_badge_text", "Standardsprache (AS-Gegenstück)"),
                    color="#2563eb", # Vibrant Blue
                    scroll_top=cfg.get("as_scroll_top", False)
                )
                await page_as.wait_for_timeout(800)
                
            as_img_path = os.path.join(src_dir, f"{source_id}_as_target.png")
            await page_as.screenshot(path=as_img_path, full_page=False)
            results["files"]["as_target"] = as_img_path
            print(f"  ✓ Saved Highlighted AS Target Screenshot: {as_img_path}")
            
            # 3. GENERATE SIDE-BY-SIDE COMPARISON
            combo_path = os.path.join(src_dir, f"{source_id}_side_by_side.png")
            if combine_side_by_side(viewport_path, as_img_path, combo_path):
                results["files"]["side_by_side"] = combo_path
                print(f"  ✓ Created Side-by-Side Comparison: {combo_path}")
                
        except Exception as e:
            print(f"  [FAIL] Error capturing AS page for {name}: {e}")
            traceback.print_exc()
        finally:
            await page_as.close()
            
    await context.close()
    return results


def generate_html_gallery(results_list, output_dir):
    """Generates an interactive HTML overview gallery to inspect all alignment screenshots."""
    html_path = os.path.join(output_dir, "index.html")
    
    cards_html = []
    for r in results_list:
        source_id = r["id"]
        name = r["name"]
        cat = r.get("category", "")
        strat = r["strategy"]
        desc = r["description"]
        ls_url = r["ls_url"]
        as_url = r.get("as_url")
        files = r.get("files", {})
        
        viewport_rel = os.path.relpath(files["viewport"], output_dir) if "viewport" in files else ""
        closeup_rel = os.path.relpath(files["closeup"], output_dir) if "closeup" in files else ""
        as_rel = os.path.relpath(files["as_target"], output_dir) if "as_target" in files else ""
        side_rel = os.path.relpath(files["side_by_side"], output_dir) if "side_by_side" in files else ""
        
        card = f"""
        <div class="source-card" id="{source_id}">
            <div class="card-header">
                <div>
                    <span class="category-badge">{cat}</span>
                    <h2 class="source-title">{name}</h2>
                </div>
                <div class="strategy-badge">{strat}</div>
            </div>
            
            <p class="description">{desc}</p>
            
            <div class="url-info">
                <div><strong>LS-Quelle:</strong> <a href="{ls_url}" target="_blank">{ls_url}</a></div>
                {f'<div><strong>AS-Gegenstück:</strong> <a href="{as_url}" target="_blank">{as_url}</a></div>' if as_url else '<div><strong>Ausrichtung:</strong> Parallele In-Page-Struktur (Gegenüberstellung auf einer Seite)</div>'}
            </div>
            
            <div class="media-grid">
                {f'''
                <div class="media-item">
                    <div class="media-label">Leichte Sprache (LS) mit hervorgehobenem Alignment-Element</div>
                    <a href="{viewport_rel}" target="_blank">
                        <img src="{viewport_rel}" alt="{name} Viewport Screenshot" loading="lazy" />
                    </a>
                </div>
                ''' if viewport_rel else ''}
                
                {f'''
                <div class="media-item">
                    <div class="media-label">Zoom / Detailansicht des Alignment-Elements</div>
                    <a href="{closeup_rel}" target="_blank">
                        <img src="{closeup_rel}" alt="{name} Element Close-up" class="closeup-img" loading="lazy" />
                    </a>
                </div>
                ''' if closeup_rel else ''}
            </div>
            
            {f'''
            <div class="side-by-side-section">
                <div class="media-label">⚖️ Paarweise Gegenüberstellung: Leichte Sprache (Links / Rot) &larr;&rarr; Standardsprache (Rechts / Blau)</div>
                <a href="{side_rel}" target="_blank">
                    <img src="{side_rel}" alt="{name} Side by Side" class="side-img" loading="lazy" />
                </a>
            </div>
            ''' if side_rel else ''}
        </div>
        """
        cards_html.append(card)
        
    full_html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scraping Alignment Visualisierungs-Report | Master Thesis</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #e11d48;
            --accent-blue: #38bdf8;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            padding: 30px 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 40px;
            text-align: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 25px;
        }}
        header h1 {{
            font-size: 2.3rem;
            color: #ffffff;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 800px;
            margin: 0 auto;
        }}
        .toc {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }}
        .toc a {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 14px;
            border-radius: 20px;
            text-decoration: none;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }}
        .toc a:hover {{
            background: var(--accent);
            color: #ffffff;
            border-color: var(--accent);
        }}
        .source-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 28px;
            margin-bottom: 35px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .source-title {{
            font-size: 1.6rem;
            color: #ffffff;
            font-weight: 600;
        }}
        .category-badge {{
            display: inline-block;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent-blue);
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .strategy-badge {{
            background: rgba(225, 29, 72, 0.15);
            color: #fb7185;
            border: 1px solid rgba(225, 29, 72, 0.4);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .description {{
            color: #cbd5e1;
            margin-bottom: 18px;
            font-size: 1.05rem;
        }}
        .url-info {{
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.9rem;
            margin-bottom: 22px;
            word-break: break-all;
        }}
        .url-info div {{ margin-bottom: 4px; }}
        .url-info div:last-child {{ margin-bottom: 0; }}
        .url-info a {{
            color: var(--accent-blue);
            text-decoration: none;
        }}
        .url-info a:hover {{ text-decoration: underline; }}
        .media-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        @media (max-width: 900px) {{
            .media-grid {{ grid-template-columns: 1fr; }}
        }}
        .media-item {{
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .media-label {{
            background: #1e293b;
            padding: 10px 14px;
            font-size: 0.88rem;
            font-weight: 600;
            color: #e2e8f0;
            border-bottom: 1px solid var(--border);
        }}
        .media-item img, .side-by-side-section img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .media-item img:hover, .side-by-side-section img:hover {{
            opacity: 0.95;
        }}
        .closeup-img {{
            object-fit: contain;
            background: #0f172a;
            padding: 15px;
            max-height: 400px;
        }}
        .side-by-side-section {{
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            margin-top: 15px;
        }}
        footer {{
            text-align: center;
            color: var(--text-muted);
            margin-top: 50px;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Web-Scraping Alignment Dokumentation</h1>
            <p>Visuelle Dokumentation der Ausrichtungs- und Matching-Mechanismen (Buttons, Links, Sprachwechsler, Parallelstrukturen) für alle 12 Korpusquellen der Masterarbeit.</p>
            <div class="toc">
                {"".join([f'<a href="#{r["id"]}">{r["name"]}</a>' for r in results_list])}
            </div>
        </header>
        
        <main>
            {"".join(cards_html)}
        </main>
        
        <footer>
            <p>Generiert für Master Thesis Corpus Creation & Alignment Verification.</p>
        </footer>
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"\nInteractive HTML Gallery generated at: {html_path}")


def generate_markdown_report(results_list, output_dir):
    """Generates a comprehensive Markdown documentation report."""
    md_path = os.path.join(output_dir, "README.md")
    
    lines = [
        "# Scraping Alignment Visualisierungs-Report",
        "",
        "Dieses Verzeichnis enthält hochauflösende Screenshots aller 12 Korpusquellen zur visuellen Erklärung des Ausrichtungs- und Extraktionsprozesses (Alignment zwischen Leichter Sprache und Standardsprache).",
        "",
        "## Übersicht der Quellen & Alignment-Strategien",
        "",
        "| Quelle | Kategorie | Alignment-Mechanismus | Screenshots |",
        "| :--- | :--- | :--- | :--- |"
    ]
    
    for r in results_list:
        name = r["name"]
        cat = r.get("category", "-")
        strat = r["strategy"]
        src_id = r["id"]
        lines.append(f"| **{name}** | {cat} | {strat} | [Zu den Bildern](#{src_id}) |")
        
    lines.append("")
    lines.append("---")
    lines.append("")
    
    for r in results_list:
        src_id = r["id"]
        name = r["name"]
        strat = r["strategy"]
        desc = r["description"]
        ls_url = r["ls_url"]
        as_url = r.get("as_url")
        files = r.get("files", {})
        
        viewport_rel = os.path.relpath(files["viewport"], output_dir) if "viewport" in files else None
        closeup_rel = os.path.relpath(files["closeup"], output_dir) if "closeup" in files else None
        as_rel = os.path.relpath(files["as_target"], output_dir) if "as_target" in files else None
        side_rel = os.path.relpath(files["side_by_side"], output_dir) if "side_by_side" in files else None
        
        lines.append(f"### {name} <a id='{src_id}'></a>")
        lines.append(f"**Strategie**: `{strat}`  ")
        lines.append(f"**Beschreibung**: {desc}  ")
        lines.append(f"- **LS URL**: [{ls_url}]({ls_url})")
        if as_url:
            lines.append(f"- **AS URL**: [{as_url}]({as_url})")
        lines.append("")
        
        if viewport_rel:
            lines.append(f"#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element")
            lines.append(f"![{name} LS Viewport]({viewport_rel})")
            lines.append("")
            
        if closeup_rel:
            lines.append(f"#### 2. Detailansicht des Alignment-Elements")
            lines.append(f"![{name} Closeup]({closeup_rel})")
            lines.append("")
            
        if as_rel:
            lines.append(f"#### 3. Standardsprachlicher Gegenstück-Artikel (AS)")
            lines.append(f"![{name} AS Target]({as_rel})")
            lines.append("")
            
        if side_rel:
            lines.append(f"#### 4. Paarweiser Vergleich (LS vs. AS)")
            lines.append(f"![{name} Gegenüberstellung]({side_rel})")
            lines.append("")
            
        lines.append("---")
        lines.append("")
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown Summary Report generated at: {md_path}")


async def main_async(args):
    from playwright.async_api import async_playwright
    
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter sources
    selected_sources = SOURCE_CONFIGS
    if args.sources and args.sources.lower() != "all":
        wanted_ids = [s.strip().lower() for s in args.sources.split(",")]
        selected_sources = [s for s in SOURCE_CONFIGS if s["id"] in wanted_ids]
        
    print(f"Running Alignment Screenshot Generator for {len(selected_sources)} source(s)...")
    print(f"Destination: {output_dir}\n")
    
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.headed)
        
        for cfg in selected_sources:
            try:
                res = await capture_source(browser, cfg, output_dir, timeout=args.timeout)
                results.append(res)
            except Exception as e:
                print(f"Error processing {cfg['name']}: {e}")
                traceback.print_exc()
                
        await browser.close()
        
    # Generate reports
    generate_html_gallery(results, output_dir)
    generate_markdown_report(results, output_dir)
    print("\n[OK] All screenshots and reports successfully created!")


def main():
    parser = argparse.ArgumentParser(description="Generate visual alignment screenshots for scraping sources.")
    parser.add_argument("--sources", "-s", type=str, default="all", help="Comma-separated source IDs or 'all' (e.g. mdr,taz,apotheken)")
    parser.add_argument("--output-dir", "-o", type=str, default="figures/scraping_alignment", help="Output directory for screenshots")
    parser.add_argument("--timeout", "-t", type=int, default=45000, help="Timeout in milliseconds per page")
    parser.add_argument("--headed", action="store_true", help="Launch browser in headed mode to see live execution")
    
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
