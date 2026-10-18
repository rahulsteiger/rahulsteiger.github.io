---
layout: default
permalink: /notes/archive/
title: Archived learning notes
nav: false
---

<div class="post notes-archive">
  <div class="header-bar">
    <h1>Archived learning notes</h1>
    <h2>Earlier course notes and experiments, preserved for reference.</h2>
  </div>

{% assign archived_notes = site.pages | where: "archived", true | sort: "date" | reverse %}

  <ul class="post-list">
    {% for note in archived_notes %}
      <li>
        <div class="archive-note-row">
          <time datetime="{{ note.date | date: '%Y-%m-%d' }}">{{ note.date | date: '%b %d, %Y' }}</time>
          <div>
            <h3><a class="post-title" href="{{ note.url | relative_url }}">{{ note.title }}</a></h3>
            <p>{{ note.description }}</p>
          </div>
        </div>
      </li>
    {% endfor %}
  </ul>
</div>
