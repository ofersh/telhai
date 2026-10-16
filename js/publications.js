/* Publications page: type filter + live search. No dependencies. */
(function () {
  document.documentElement.classList.add('js');
  var pills = [].slice.call(document.querySelectorAll('.pill'));
  var items = [].slice.call(document.querySelectorAll('.pub'));
  var groups = [].slice.call(document.querySelectorAll('.year'));
  var input = document.getElementById('pub-q');
  var count = document.getElementById('pub-count');
  var empty = document.getElementById('pub-empty');
  var kind = 'pubs';

  function matches(filter, k) { return filter === 'pubs' ? k !== 'p' : k === filter; }

  // Each button's count comes from the list itself, so it stays right after edits.
  pills.forEach(function (b) {
    var f = b.getAttribute('data-kind');
    b.querySelector('.pill__n').textContent = items.filter(function (li) { return matches(f, li.getAttribute('data-kind')); }).length;
    b.addEventListener('click', function () { kind = f; apply(); });
  });

  function apply() {
    var q = input.value.trim().toLowerCase();
    var shown = 0, total = 0;
    items.forEach(function (li) {
      var ok = matches(kind, li.getAttribute('data-kind'));
      if (ok) total++;
      var text = (li.textContent + ' ' + li.parentNode.parentNode.getAttribute('data-label')).toLowerCase();
      var hit = ok && (!q || text.indexOf(q) >= 0);
      li.hidden = !hit;
      if (hit) shown++;
    });
    groups.forEach(function (g) { g.hidden = !g.querySelector('.pub:not([hidden])'); });
    pills.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-kind') === kind)); });
    count.textContent = shown + ' of ' + total + ' shown';
    empty.hidden = shown > 0;
  }

  input.addEventListener('input', apply);
  apply();
})();
