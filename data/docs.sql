-- Initial docs seed data
-- This file contains INSERT statements for initial documentation

INSERT INTO w4h.docs (slug, title, card_title, description, action_label, doc_type, sort_order, hidden, is_feature_card, show_in_docs, card_order, md, html, updated_at)
VALUES 
  (
    'home',
    'Welcome to W4H',
    'Home',
    'Welcome to the W4H Toolkit',
    'Open',
    '',
    0,
    false,
    false,
    true,
    0,
    '# Welcome to the W4H

Welcome to the W4H Toolkit. This is your starting point for managing data and creating insights.',
    '<h1>Welcome to W4H</h1>
<p>Welcome to the W4H Toolkit. This is your starting point for managing data and creating insights.</p>',
    CURRENT_TIMESTAMP
  ),
  (
    'quick-start',
    'Quick Start Guide',
    'Quick Start',
    'Get started with W4H in 5 minutes',
    'Open',
    'guide',
    1,
    false,
    true,
    true,
    1,
    '# Quick Start Guide

## Getting Started

Follow these steps to get started with W4H:

1. Configure your database connection
2. Create your first dataset
3. Explore the data tools
4. Create a visualization
5. Share your insights',
    '<h1>Quick Start Guide</h1>
<h2>Getting Started</h2>
<p>Follow these steps to get started with W4H:</p>
<ol>
<li>Configure your database connection</li>
<li>Create your first dataset</li>
<li>Explore the data tools</li>
<li>Create a visualization</li>
<li>Share your insights</li>
</ol>',
    CURRENT_TIMESTAMP
  ),
  (
    'faq',
    'Frequently Asked Questions',
    'FAQ',
    'Common questions and answers',
    'Open',
    'faq',
    10,
    false,
    false,
    true,
    0,
    '# Frequently Asked Questions

## What is W4H?

W4H is a toolkit for managing data and creating insights from neighborhood and city-level data.

## How do I get started?

See the Quick Start guide for step-by-step instructions.

## Where can I find more documentation?

Browse the docs section for comprehensive guides and references.',
    '<h1>Frequently Asked Questions</h1>
<h2>What is W4H?</h2>
<p>W4H is a toolkit for managing data and creating insights from neighborhood and city-level data.</p>
<h2>How do I get started?</h2>
<p>See the Quick Start guide for step-by-step instructions.</p>
<h2>Where can I find more documentation?</h2>
<p>Browse the docs section for comprehensive guides and references.</p>',
    CURRENT_TIMESTAMP
  )
ON CONFLICT (slug) DO NOTHING;
