---
layout: default
permalink: /notes/
title: Notes
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 10
  sort_field: date
  sort_reverse: true
  trail:
    before: 1
    after: 3
---

<div class="post">

{% assign blog_name_size = site.blog_name | size %}
{% assign blog_description_size = site.blog_description | size %}

{% if blog_name_size > 0 or blog_description_size > 0 %}

  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    <h2>{{ site.blog_description }}</h2>
  </div>
{% endif %}

  <ul class="post-list" style="list-style: none; padding: 0;">

    {% if page.pagination.enabled %}
      {% assign postlist = paginator.posts %}
    {% else %}
      {% assign postlist = site.posts %}
    {% endif %}

    {% for post in postlist %}
    <li style="margin-bottom: 1.5rem;">
      <div style="display: flex; align-items: baseline; gap: 3rem;">
        <div style="color: var(--global-text-color-light); font-size: 1.55rem; white-space: nowrap; flex-shrink: 0;">
          {{ post.date | date: '%b %d, %Y' }}
        </div>
        <div style="flex: 1;">
          <h3 style="margin: 0 0 0.25rem 0; font-size: 1.55rem;">
            {% if post.redirect == blank %}
              <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
            {% elsif post.redirect contains '://' %}
              <a class="post-title" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
            {% else %}
              <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
            {% endif %}
          </h3>
          {% if post.description %}
            <p style="margin: 0; color: var(--global-text-color-light); font-size: 1.2rem;">{{ post.description }}</p>
          {% endif %}
        </div>
      </div>
    </li>
    {% endfor %}

  </ul>

{% if page.pagination.enabled %}
{% include pagination.liquid %}
{% endif %}

</div>
