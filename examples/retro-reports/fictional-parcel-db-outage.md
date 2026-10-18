---
title: "Fictional postmortem: order database outage at Example Parcel Co."
source_url: null
organisation: Example Parcel Co. (fictional)
published: 2026-10-17
license: CC0 (written for the JETTAE test fixture; not a real organisation or incident)
---
# Postmortem: order database outage (fictional)

## Summary

On a Saturday night the primary order database stopped accepting writes for 2 hours 40 minutes.
Customers could not create shipments during that time.

## Response set-up before the incident

The order database runbook told the primary on-call engineer to page the database team in the
#db-oncall Slack channel when the write error rate alarm fired. The runbook named Dana Kim as the
owner of the order database. There was no secondary on-call for the database team. Failing over to
the replica required approval from the CTO. The runbook did not describe how to roll back a schema
migration.

## Timeline

- 22:05 The write error rate alarm fired and paged the primary on-call engineer.
- 22:10 The on-call engineer posted in #db-oncall, but Slack was unavailable because it was routed
  through the same network segment that failed.
- 22:55 The on-call engineer reached Dana Kim by personal phone after searching the HR directory.
- 23:20 The team decided to fail over, but the CTO could not be reached for approval because he was
  on a flight. Nobody else was allowed to approve.
- 00:30 The CTO approved the failover after landing.
- 00:45 Writes recovered.

## What went wrong in the response

- The only channel the runbook gave for reaching the database team depended on the failed network.
- Nobody was designated to take over when the primary on-call could not reach the database owner.
- Only one person could approve a failover, and no deputy existed.
- There was no documented rollback for the schema migration that triggered the outage.

## Root cause

A schema migration locked the orders table and exhausted the connection pool.

## Follow-up actions

- Add a phone bridge as a fallback channel for the database team.
- Create a secondary on-call rotation.
- Allow the VP of Engineering to approve failovers.
- Document the migration rollback procedure.
