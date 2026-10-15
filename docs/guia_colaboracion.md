# Guía de colaboración con Git

Cada integrante trabaja en **su propia rama** y `main` solo recibe lo revisado.

| Rama | Responsable |
|---|---|
| `main` | versión final del equipo |
| `CARLOS-BEDOYA` | esta versión (código comentado + guion de exposición) |
| `Sebastian/Planteamiento`, `Julian_Branch`, `Santi` | ramas de los demás integrantes |

## Flujo diario
```bash
git checkout CARLOS-BEDOYA          # trabajar siempre en la rama propia
git pull                            # traer lo último de GitHub
# ... editar ...
git add -A
git commit -m "feat: descripción corta del cambio"
git push                            # subir la rama
```
Para llevar la rama a `main`: abrir un **Pull Request** en GitHub (`CARLOS-BEDOYA → main`) y que otro integrante lo revise.

## Qué NO se sube
`data/raw/` (cada CSV pesa ~80 MB), `data/processed/`, `models/*.joblib`, entornos virtuales y cachés (ver `.gitignore`).
Sí se suben las tablas pequeñas de evidencia: `reports/*.csv`, `models/*.csv`, `models/eleccion.txt`.

## Antes de hacer push del notebook
`Kernel → Restart & Run All` con los datos reales, para que las salidas visibles en GitHub sean las de la corrida oficial (semilla 42).
